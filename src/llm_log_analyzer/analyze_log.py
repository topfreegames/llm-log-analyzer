#!/usr/bin/env python3
"""
Log Analyzer

A tool for analyzing log files to identify root causes of issues.
Designed for CI environments (Jenkins, GitLab) with large log processing capabilities.

"""

import os
import sys
import json
import logging
import argparse
import tiktoken
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from llm_log_analyzer.providers import create_llm_provider
from llm_log_analyzer.llm_client import AnalysisResult, LLMClient, resolve_failure_class_fallback
from llm_log_analyzer.environment import get_provider_status
from llm_log_analyzer.utils import get_confidence_emoji, load_list_from_file, load_string_from_file, MIN_CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
from llm_log_analyzer.constants import *
from llm_log_analyzer.presets import DEFAULT_PRESET, get_preset, is_valid_preset, list_presets
from langchain_text_splitters import TokenTextSplitter

class LogProcessor:
    """Handles log file processing, filtering, and chunking."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, filter_patterns: List[str] = [], filter_keywords_regex: re.Pattern = None,
                 tokenizer_encoding_name: str = DEFAULT_TOKENIZER):
        self.logger = logger or logging.getLogger(__name__)
        self.tokenizer_encoding_name = tokenizer_encoding_name
        self.encoding = tiktoken.get_encoding(tokenizer_encoding_name)
        self.filter_keywords_regex = filter_keywords_regex or re.compile(r'(?!)')
        self.filter_patterns = filter_patterns
        
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(file_path)
    
    def extract_tail_lines(self, file_path: str, num_lines: int) -> List[str]:
        """
        Extract the last N lines from a file efficiently.
        
        Args:
            file_path: Path to the log file
            num_lines: Number of lines to extract from the end
            
        Returns:
            List of lines from the end of the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                
                # For large files, seek to end and read backwards efficiently
                file_size = self.get_file_size(file_path)
                if file_size > 50 * 1024 * 1024:  # 50MB threshold
                    self.logger.info("Large file detected, using optimized tail extraction...")
                    return self._extract_tail_large_file(f, num_lines)
                else:
                    # For smaller files, read all lines and return the tail
                    lines = f.readlines()
                    return lines[-num_lines:] if len(lines) > num_lines else lines
                    
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return []
    
    def _extract_tail_large_file(self, file_handle, num_lines: int) -> List[str]:
        """Extract tail lines from very large files using seek."""
        lines = []
        buffer_size = 8192
        file_handle.seek(0, 2)  # Go to end
        file_size = file_handle.tell()
        
        position = file_size
        line_count = 0
        
        while position > 0 and line_count < num_lines:
            read_size = min(buffer_size, position)
            position -= read_size
            
            file_handle.seek(position)
            chunk = file_handle.read(read_size)
            
            chunk_lines = chunk.split('\n')
            if position == 0:
                lines = chunk_lines + lines
            else:
                lines = chunk_lines[1:] + lines
                
            line_count = len(lines)
        
        return lines[-num_lines:] if len(lines) > num_lines else lines
    
    def filter_relevant_lines(self, lines: List[str], context_lines: int = 100) -> List[str]:
        """
        Filter lines that contain specific patterns and include context around them.
        
        Args:
            lines: List of log lines
            context_lines: Number of lines to include before and after each relevant log
            
        Returns:
            Filtered list of relevant lines with context and block separators
        """
        if not lines:
            return []
            
        relevant_indices = []
        
        # First pass: identify all relevant line indices
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            if (self.filter_keywords_regex.search(line_stripped) or 
                any(pattern in line_stripped for pattern in self.filter_patterns)):
                relevant_indices.append(i)
        
        if not relevant_indices:
            return []
        
        # Create context ranges around each relevant line
        ranges = []
        for relevant_idx in relevant_indices:
            start = max(0, relevant_idx - context_lines)
            end = min(len(lines), relevant_idx + context_lines + 1)
            ranges.append((start, end))
        
        # Merge overlapping ranges
        if ranges:
            ranges.sort()
            merged_ranges = [ranges[0]]
            
            for start, end in ranges[1:]:
                last_start, last_end = merged_ranges[-1]
                if start <= last_end:  # Overlapping or adjacent ranges
                    merged_ranges[-1] = (last_start, max(last_end, end))
                else:
                    merged_ranges.append((start, end))
        else:
            merged_ranges = []
        
        filtered = []
        total_lines_added = 0
        
        for i, (start, end) in enumerate(merged_ranges):
            if i > 0:
                filtered.append("--------- RELEVANT BLOCK SEPARATOR ---------")
            
            block_lines = 0
            for line_idx in range(start, end):
                filtered.append(lines[line_idx].rstrip())
                block_lines += 1
            
            total_lines_added += block_lines
        
        self.logger.debug(f"Found {len(relevant_indices)} relevant lines, created {len(merged_ranges)} context blocks, total filtered lines: {total_lines_added}")        
        return filtered
    
    def save_filtered_log(self, lines: List[str], output_path: str) -> None:
        """Save filtered log lines to a file for debugging."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            self.logger.debug(f"Saved filtered log to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save filtered log: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """
        Split text into chunks suitable for LLM processing using token-based sizing.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk
            
        Returns:
            List of text chunks with token-aware overlap
        """
        if not text.strip():
            return []
        
        text_splitter = TokenTextSplitter(
            encoding_name=self.tokenizer_encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=DEFAULT_OVERLAP_SIZE
        )
        
        return text_splitter.split_text(text)
            
class OutputGenerator:
    """Handles generation of JSON and Markdown outputs."""
    
    def __init__(self, output_dir: str = ".", logger: Optional[logging.Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def save_json_analysis(self, result: AnalysisResult, file_name: str = JSON_OUTPUT_FILE) -> str:
        """Save analysis result as JSON."""
        output_path = self.output_dir / file_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved JSON analysis to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON analysis: {e}")
            return ""
    
    def save_markdown_summary(self, result: AnalysisResult, analyzer: 'LogAnalyzer' = None, file_name: str = MARKDOWN_OUTPUT_FILE) -> str:
        """Save human-friendly analysis summary as Markdown."""
        output_path = self.output_dir / file_name
        
        try:
            content = self._generate_markdown_content(result, analyzer)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Saved Markdown summary to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save Markdown summary: {e}")
            return ""
    
    def _generate_markdown_content(self, result: AnalysisResult, analyzer: 'LogAnalyzer' = None) -> str:
        """Generate Markdown content for the analysis result."""
        confidence_emoji = get_confidence_emoji(result.confidence)
        
        failure_class = getattr(result, "failure_class", "")
        failure_class_section = (
            f"\n**Failure Class:** `{failure_class}`\n" if failure_class else ""
        )
        
        provider_section = ""
        if analyzer and analyzer.llm_client:
            try:
                provider_info = analyzer.llm_client.get_provider_info()
                provider_section = f"""
## Analysis Configuration

**Provider:** {provider_info['provider']}

**Models Used:** 
- Chunk Analysis: {provider_info['default_chunk_model']}
- Final Aggregation: {provider_info['default_aggregation_model']}

**Processing Settings:**
- Max Chunks: {analyzer.max_chunks}
- Max Workers: {analyzer.max_workers}
"""
            except Exception:
                provider_section = ""
        
        content = f"""# Build Analysis Report

## Summary
{result.human_summary}
{provider_section}
## Analysis Details

**Root Cause:** {result.root_cause}
{failure_class_section}

**Confidence:** {confidence_emoji} {result.confidence:.1%}

## Action Items
{result.action_suggestion}

## Relevant Log Lines
"""
        
        if result.relevant_lines:
            content += "```\n"
            for line in result.relevant_lines:
                content += f"{line}\n"
            content += "```\n"
        else:
            content += "*No specific log lines identified*\n"
        
        content += """
---
*Generated by Log Analyzer*
"""
        
        return content
    
class LogAnalyzer:
    """Main analyzer class that orchestrates the entire analysis process."""
    
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None, output_dir: str = DEFAULT_OUTPUT_DIR, verbose: bool = False, quiet: bool = False, debug: bool = False, max_chunks: int = DEFAULT_MAX_CHUNKS, max_workers: int = DEFAULT_MAX_PARALLEL_CHUNKS, context_lines: int = DEFAULT_CONTEXT_LINES, chunk_size: int = DEFAULT_CHUNK_SIZE, filter_keywords: List[str] = DEFAULT_FILTER_KEYWORDS, filter_regex: Optional[str] = None, preset: Optional[str] = None, patterns_file: str = None, prompt: Optional[str] = None, prompt_file: Optional[str] = None, additional_context_file: str = None, chunk_model: Optional[str] = None, aggregation_model: Optional[str] = None, timeout: Optional[float] = None, failure_classes: Optional[Dict[str, str]] = None, failure_class_fallback: Optional[str] = None):
        """
            Initialize the log analyzer.
            
            Args:
                provider: LLM provider to use
                api_key: API key for the selected provider
                output_dir: Directory for output files
                verbose: Enable verbose logging
                quiet: Suppress info logs (only show warnings and errors)
                debug: Enable debug files
                max_chunks: Maximum number of chunks to analyze
                max_workers: Maximum parallel workers for chunk analysis
                context_lines: Number of lines to include before and after each relevant line
                chunk_size: Size of chunks in bytes for large log processing
                filter_keywords: List of keywords to filter log lines
                filter_regex: Custom regex pattern to filter log lines (combined with filter_keywords via OR)
                preset: Preset configuration - takes precedence over patterns_file/prompt_file
                patterns_file: Path to file containing specific patterns to look for
                prompt: Raw prompt intro text (takes precedence over prompt_file and preset)
                prompt_file: Path to prompt intro file (used if prompt is not provided)
                additional_context_file: Path to file containing additional context for known issues and solutions
                chunk_model: Model to use for chunk analysis (if not provided, uses provider default)
                aggregation_model: Model to use for aggregation analysis (if not provided, uses provider default)
                timeout: Timeout in seconds for API requests (None means no timeout)
                failure_classes: Closed set of failure-class identifiers for the model to choose
                    from, as bare identifiers or identifier -> description
                failure_class_fallback: Which of those identifiers to record when the model's
                    answer is unusable (default: "unknown", when the set contains it)
        """
        # Determine log level: verbose > quiet > default
        if verbose:
            log_level = logging.DEBUG
        elif quiet:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        self.quiet = quiet
        self.debug_output_dir = Path(output_dir) if debug else None
        
        try:
            self.llm_provider = create_llm_provider(provider=provider, api_key=api_key, timeout=timeout)
            self.llm_client = LLMClient(provider=self.llm_provider, debug_output_dir=self.debug_output_dir, chunk_model=chunk_model, aggregation_model=aggregation_model, failure_classes=failure_classes, failure_class_fallback=failure_class_fallback)
                        
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise e
        
        # Resolve preset, patterns, and prompts with proper precedence
        resolved_patterns_file, resolved_prompt_file = self._resolve_preset_and_files(preset, patterns_file, prompt_file)
        filter_patterns = self._resolve_patterns(resolved_patterns_file)
        filter_keywords_regex = self._build_keywords_regex(filter_keywords, filter_regex)

        self.log_processor = LogProcessor(logger=self.logger, filter_patterns=filter_patterns, filter_keywords_regex=filter_keywords_regex, tokenizer_encoding_name=self.llm_provider.get_default_tokenizer_encoding_name() or DEFAULT_TOKENIZER)
        self.output_generator = OutputGenerator(output_dir=output_dir, logger=self.logger)
        self.max_chunks = max_chunks
        self.max_workers = max_workers
        self.context_lines = context_lines
        self.chunk_size = chunk_size
        self.additional_context = self._load_additional_context(additional_context_file)
        
        self.prompt = self._resolve_prompt(prompt, resolved_prompt_file)
        
        self.logger.info(f"Log Analyzer initialized (max_chunks: {max_chunks}, max_workers: {max_workers}, context_lines: {context_lines}, chunk_size: {chunk_size} tokens)")
    
        self.logger.debug(f"Using tokenizer: {self.log_processor.tokenizer_encoding_name}, patterns: {resolved_patterns_file}, prompt: {self.prompt[:50]}...")
        self.logger.debug(f"Configured {len(filter_keywords)} filter keywords, filter_regex={filter_regex!r}, and {len(filter_patterns)} filter patterns for log processor")

    def _resolve_preset_and_files(self, preset: Optional[str], patterns_file: Optional[str], prompt_file: Optional[str]) -> Tuple[str, str]:
        """
        Resolve preset, patterns, and prompt files with proper precedence:
        1. Explicit patterns_file/prompt_file parameters take highest precedence
        2. Preset parameter takes second precedence  
        3. Fall back to default preset if nothing specified
        
        Returns:
            Tuple of (resolved_patterns_file, resolved_prompt_file)
        """
        # Use default preset if nothing specified
        effective_preset = preset or DEFAULT_PRESET
        
        # Validate preset exists
        if not is_valid_preset(effective_preset):
            available_presets = ", ".join(list_presets().keys())
            raise ValueError(f"Invalid preset '{effective_preset}'. Available presets: {available_presets}")
        
        preset_config = get_preset(effective_preset)
        
        # Resolve patterns file: explicit parameter > preset > default
        resolved_patterns_file = patterns_file or preset_config["patterns_file"]
        
        # Resolve prompt file: explicit parameter > preset > default  
        resolved_prompt_file = prompt_file or preset_config["prompt_file"]
        
        if preset:
            self.logger.info(f"Using preset '{effective_preset}': {preset_config['description']}")
        
        self.logger.debug(f"Resolved files - patterns: {resolved_patterns_file}, prompt: {resolved_prompt_file}")
        
        return resolved_patterns_file, resolved_prompt_file

    def _resolve_prompt(self, prompt: Optional[str], prompt_file: Optional[str]) -> Optional[str]:
        """Resolve prompt precedence: inline text takes precedence over file."""
        if prompt:
            self.logger.debug(f"Using inline prompt text: {prompt[:50]}...")
            return prompt
        
        if prompt_file:
            resolved_prompt = load_string_from_file(prompt_file)
            self.logger.debug(f"Loaded prompt from {prompt_file}: {resolved_prompt[:50]}...")
            return resolved_prompt
        
        raise Exception("No prompt specified")
    
    def _resolve_patterns(self, patterns_file: Optional[str]) -> List[str]:
        """Resolve patterns from file."""
        if patterns_file:
            resolved_patterns = load_list_from_file(patterns_file)
            self.logger.debug(f"Loaded patterns from {patterns_file}: {patterns_file[:50]}...")
            return resolved_patterns
        
        return []
    
    def _build_keywords_regex(self, keywords, filter_regex: Optional[str] = None):
        """Build combined regex from keyword list and optional raw regex pattern."""
        parts = []
        
        clean_keywords = [k for k in keywords if k and k.strip()]
        if clean_keywords:
            parts.append(r'\b(' + '|'.join(clean_keywords) + r')\b')
        
        if filter_regex:
            re.compile(filter_regex)
            parts.append(filter_regex)
        
        if not parts:
            return None
        return re.compile('|'.join(parts), re.IGNORECASE)

    def _load_additional_context(self, additional_context_file: str) -> str:
        """Load additional context from file if provided."""
        if not additional_context_file or not additional_context_file.strip():
            return ''
        
        try:
            if not os.path.exists(additional_context_file):
                raise Exception(f"Additional context file not found: {additional_context_file}")
            
            with open(additional_context_file, 'r', encoding='utf-8', errors='replace') as f:
                context = f.read().strip()
                if context:
                    self.logger.info(f"Loaded additional context from {additional_context_file} ({len(context)} characters)")
                    return context
                else:
                    self.logger.info(f"Additional context file is empty: {additional_context_file}")
                    return ''
                    
        except Exception as e:
            raise Exception(f"Error loading additional context from {additional_context_file}: {e}")
    
    def analyze_log_file(self, log_path: str, tail_lines: int = DEFAULT_TAIL_LINES) -> Tuple[AnalysisResult, bool]:
        """
        Analyze a log file and determine the root cause of failure.
        
        Args:
            log_path: Path to the log file
            tail_lines: Number of lines to extract from the end of the log
            
        Returns:
            Tuple of (AnalysisResult, success_flag)
        """
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file not found: {log_path}")
        
        file_size = self.log_processor.get_file_size(log_path)
        self.logger.info(f"Analyzing log file: {log_path} ({file_size / 1024 / 1024:.1f} MB)")
        
        self.logger.debug(f"Extracting last {tail_lines} lines...")
        tail_lines_list = self.log_processor.extract_tail_lines(log_path, tail_lines)
        
        if not tail_lines_list:
            self.logger.error("No lines extracted from log file")
            return self._create_failure_result("Unable to read log file"), False
        
        self.logger.debug(f"Extracted {len(tail_lines_list)} lines")
        
        self.logger.debug(f"Filtering relevant lines with ±{self.context_lines} lines of context...")
        filtered_lines = self.log_processor.filter_relevant_lines(tail_lines_list, context_lines=self.context_lines)
        
        if not filtered_lines:
            self.logger.warning("No filtered lines found.")
            return self._create_success_result(), False
        
        self.logger.info(f"Filtered {len(filtered_lines)} relevant lines")
        
        filtered_text = '\n'.join(filtered_lines)
        if self.debug_output_dir:
            self.log_processor.save_filtered_log(filtered_lines, str(self.debug_output_dir / FILTERED_LOG_FILE))
        
        log_tokens = len(self.log_processor.encoding.encode(filtered_text))
        self.logger.debug(f"Token count: {log_tokens} tokens")
        
        if log_tokens >= 1.5 * self.chunk_size:
            self.logger.info(f"Filtered log is large ({log_tokens} tokens), chunking for analysis...")
            return self._analyze_with_chunking(filtered_text, max_chunks=self.max_chunks)
        else:
            self.logger.info(f"Filtered log is manageable ({log_tokens} tokens), analyzing directly...")
            return self._analyze_directly(filtered_text)
    
    def _analyze_with_chunking(self, filtered_text: str, max_chunks: int = DEFAULT_MAX_CHUNKS) -> Tuple[AnalysisResult, bool]:
        """Analyze large filtered text using chunking approach."""
        chunks = self.log_processor.chunk_text(filtered_text, chunk_size=self.chunk_size)

        total_tokens = len(self.log_processor.encoding.encode(filtered_text))
        avg_chunk_tokens = total_tokens // len(chunks) if chunks else 0
        
        self.logger.debug(f"Chunking text: ~{total_tokens} tokens total, "
                        f"{len(chunks)} chunks, ~{avg_chunk_tokens} tokens/chunk avg")
        
        original_chunk_count = len(chunks)
        
        if len(chunks) > max_chunks:
            self.logger.warning(f"Generated {original_chunk_count} chunks, limiting to {max_chunks} to control API costs")
            self.logger.info("Taking first and last chunks to capture both start and end of relevant lines")
            
            if max_chunks >= 2:
                # Take first half of limit from beginning, second half from end
                first_chunks = max_chunks // 2
                last_chunks = max_chunks - first_chunks
                chunks = chunks[:first_chunks] + chunks[-last_chunks:]
            else:
                chunks = chunks[:max_chunks]
            
            self.logger.info(f"Using {len(chunks)} chunks for analysis (reduced from {original_chunk_count})")
        else:
            self.logger.info(f"Using {len(chunks)} chunks for analysis")
        
        chunk_summaries = self._analyze_chunks_parallel(chunks)
        
        if not chunk_summaries:
            return self._create_failure_result("All chunk analyses failed"), False
        
        self.logger.info("Aggregating chunk summaries for final analysis...")
        try:
            result = self.llm_client.aggregate_analysis(
                chunk_summaries, 
                custom_prompt=self.prompt,
                additional_context=self.additional_context
            )
            
            if original_chunk_count > max_chunks:
                result.human_summary = f"[Analyzed {len(chunks)}/{original_chunk_count} chunks due to limits] " + result.human_summary
            
            return result, result.confidence >= MIN_CONFIDENCE_THRESHOLD
        except Exception as e:
            self.logger.error(f"Failed to aggregate analysis: {e}")
            return self._create_failure_result(f"Aggregation failed: {str(e)}"), False
    
    def _analyze_chunks_parallel(self, chunks: List[str]) -> List[str]:
        """Analyze chunks in parallel using ThreadPoolExecutor."""
        max_workers = min(self.max_workers, len(chunks))
        
        if len(chunks) == 1:
            self.logger.info("Analyzing single chunk...")
            try:
                summary = self.llm_client.analyze_chunk(chunks[0], 0, custom_prompt=self.prompt, additional_context=self.additional_context)
                return [summary]
            except Exception as e:
                self.logger.error(f"Failed to analyze single chunk: {e}")
                return [f"Analysis failed: {str(e)}"]
        
        self.logger.info(f"Analyzing {len(chunks)} chunks in parallel (max workers: {max_workers})...")
        start_time = time.time()
        
        chunk_summaries = [None] * len(chunks)
        failed_chunks = []
        
        def analyze_single_chunk(chunk_data):
            """Analyze a single chunk - helper function for parallel execution."""
            chunk_index, chunk_content = chunk_data
            try:
                summary = self.llm_client.analyze_chunk(chunk_content, chunk_index, custom_prompt=self.prompt, additional_context=self.additional_context)
                self.logger.debug(f"Chunk {chunk_index + 1} completed: {summary[:100]}...")
                return chunk_index, summary, None
            except Exception as e:
                error_msg = f"Analysis failed for chunk {chunk_index + 1}: {str(e)}"
                self.logger.error(error_msg)
                return chunk_index, error_msg, e
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_data = [(i, chunk) for i, chunk in enumerate(chunks)]
            future_to_index = {
                executor.submit(analyze_single_chunk, data): data[0] 
                for data in chunk_data
            }
            
            completed_count = 0
            for future in as_completed(future_to_index):
                chunk_index, result, error = future.result()
                chunk_summaries[chunk_index] = result
                completed_count += 1
                
                if error is None:
                    self.logger.info(f"Completed chunk {chunk_index + 1}/{len(chunks)} ({completed_count}/{len(chunks)} total)")
                else:
                    failed_chunks.append(chunk_index + 1)
        
        elapsed_time = time.time() - start_time
        success_count = len(chunks) - len(failed_chunks)
        self.logger.info(f"Parallel analysis completed in {elapsed_time:.1f}s: {success_count}/{len(chunks)} chunks successful")
        
        if failed_chunks:
            self.logger.warning(f"Failed chunks: {failed_chunks}")
        
        final_summaries = [summary for summary in chunk_summaries if summary is not None]
        
        return final_summaries
    
    def _analyze_directly(self, filtered_text: str) -> Tuple[AnalysisResult, bool]:
        """Analyze filtered text directly without chunking."""
        try:
            result = self.llm_client.aggregate_analysis(
                [filtered_text],
                custom_prompt=self.prompt,
                additional_context=self.additional_context
            )
            return result, result.confidence >= MIN_CONFIDENCE_THRESHOLD
        except Exception as e:
            self.logger.error(f"Direct analysis failed: {e}")
            return self._create_failure_result(f"Direct analysis failed: {str(e)}"), False
    
    def _create_failure_result(self, message: str) -> AnalysisResult:
        """Create a failure result."""
        return AnalysisResult(
            root_cause=message,
            relevant_lines=[],
            confidence=0.0,
            action_suggestion="Review the log file manually or check analyzer configuration",
            human_summary=f"Analysis failed: {message}"
        )
    
    def _create_success_result(self) -> AnalysisResult:
        """Create a result for no failures detected in the log."""
        return AnalysisResult(
            root_cause="No failures detected",
            relevant_lines=[],
            confidence=HIGH_CONFIDENCE_THRESHOLD,
            action_suggestion="Log appears to indicate successful completion",
            human_summary="No significant logs found in the file. The process appears to have completed successfully."
        )
    
    def save_results(self, result: AnalysisResult) -> Tuple[str, str]:
        """
        Save analysis results to JSON and Markdown files.
        
        Args:
            result: Analysis result to save
            
        Returns:
            Tuple of (json_path, markdown_path)
        """
        json_path = self.output_generator.save_json_analysis(result)
        markdown_path = self.output_generator.save_markdown_summary(result, self)
        return json_path, markdown_path


def format_analysis_summary(result: AnalysisResult, provider_info: dict, markdown_path: str) -> str:
    """Format analysis results into a human-readable summary string."""
    confidence_emoji = get_confidence_emoji(result.confidence)
    
    summary_parts = []
    summary_parts.append(f"Root Cause: {result.root_cause}")
    if getattr(result, "failure_class", ""):
        summary_parts.append(f"Failure Class: {result.failure_class}")
    summary_parts.append(f"Confidence: {confidence_emoji} {result.confidence:.1%}")
    summary_parts.append(f"Provider: {provider_info['provider']}")
    
    if result.relevant_lines:
        summary_parts.append("\nKey Log Lines:")
        for line in result.relevant_lines:
            summary_parts.append(f"  > {line}")
    
    summary_parts.append("\n" + result.human_summary)
    summary_parts.append(f"\n💡 For resolution tips and detailed analysis, see: {markdown_path}")
    
    return "\n".join(summary_parts)

def save_analysis_summary(summary: str, summary_file_path: Path, logger: Optional[logging.Logger] = None) -> None:
    """Save the analysis summary to a file."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.debug(f"Analysis summary saved to: {summary_file_path}")
    except Exception as e:
        logger.error(f"Failed to save analysis summary to {summary_file_path}: {e}")

def load_failure_class_config(spec: Optional[str], explicit_fallback: Optional[str]
                             ) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str]]:
    """Resolve --failure-classes / --failure-class-fallback into (classes, fallback, warning).

    Never fatal. Classification is an addition to the analysis, not a precondition
    for it: a missing enum file or an enum with no usable fallback disables the
    class and leaves the log analysis - the part developers actually read - intact.
    Callers that need to know a class was requested and not produced read the empty
    `failure_class`, which is what "no classification was produced" already means.

    Deliberately absent: choosing an arbitrary member as the fallback. Recording a
    real class when the model meant "no idea" misattributes the failure, which is
    worse than recording nothing.
    """
    if not spec or not spec.strip():
        if explicit_fallback:
            return None, None, ("--failure-class-fallback was given without --failure-classes "
                                "to pick from; no classification will be requested")
        return None, None, None

    spec = spec.strip()
    if spec.startswith('@'):
        class_path = spec[1:]
        try:
            with open(class_path, 'r', encoding='utf-8') as f:
                raw = [line.split('#', 1)[0].strip() for line in f]
        except OSError as e:
            return None, None, (f"could not read --failure-classes file {class_path}: {e}; "
                                "continuing without a failure classification")
    else:
        raw = spec.split(',')

    # "identifier" or "identifier: what it means". The description is optional per
    # entry, so a bare list still works.
    classes: Dict[str, str] = {}
    for entry in raw:
        identifier, _, description = entry.strip().partition(':')
        identifier = identifier.strip()
        if identifier:
            classes[identifier] = description.strip()
    if not classes:
        return None, None, ("--failure-classes resolved to an empty list; continuing "
                            "without a failure classification")

    try:
        fallback = resolve_failure_class_fallback(classes, explicit_fallback)
    except ValueError as e:
        return None, None, f"{e}; continuing without a failure classification"

    return classes, fallback, None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze log files to identify failure causes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses generic preset by default)
  llm-log-analyzer application.log
  
  # Use Unity preset for Unity build logs
  llm-log-analyzer build.log -P unity
  
  # Use generic preset explicitly
  llm-log-analyzer server.log -P generic
  
  # Advanced options with presets
  llm-log-analyzer build.log -P unity --tail-lines 5000 --output-dir results/
  llm-log-analyzer server.log -P generic --provider openai --api-key your_openai_key
  
  # Override preset files individually if needed
  llm-log-analyzer xcode.log -P unity --prompt "You are a Unity expert focusing on memory issues..."
  llm-log-analyzer build.log --patterns-file custom_patterns.txt --prompt-file custom_prompt.txt
  
  # Usage without presets (manual file specification)
  llm-log-analyzer build.log --patterns-file patterns/generic.txt --prompt-file prompts/generic.txt
  
  # Provider and model customization
  llm-log-analyzer build.log -P unity --provider openai --debug
  llm-log-analyzer build.log -P generic --provider gemini --debug
  llm-log-analyzer build.log -P unity --provider claude --debug
  llm-log-analyzer system.log -P generic --max-chunks 5 --verbose
  llm-log-analyzer build.log -P unity --chunk-model gpt-5-mini --aggregation-model gpt-5 --provider openai
  llm-log-analyzer server.log -P generic --timeout 60 --provider gemini
        """
    )
    
    parser.add_argument(
        'log', 
        nargs='?',
        help='Path to the log file (not required when using --list-providers)'
    )
    
    parser.add_argument(
        '-p', '--provider',
        choices=['claude', 'gemini', 'openai', 'aws-bedrock-claude'],
        default=DEFAULT_PROVIDER,
        help=f'LLM provider to use (default: {DEFAULT_PROVIDER})'
    )
    
    parser.add_argument(
        '-t', '--tail-lines', 
        type=int, 
        default=DEFAULT_TAIL_LINES,
        help=f'Number of lines to extract from the end of the log (default: {DEFAULT_TAIL_LINES})'
    )
    
    parser.add_argument(
        '-c', '--max-chunks',
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help=f'Maximum number of chunks to analyze (default: {DEFAULT_MAX_CHUNKS})'
    )
    
    parser.add_argument(
        '-w', '--max-workers',
        type=int,
        default=DEFAULT_MAX_PARALLEL_CHUNKS,
        help=f'Maximum parallel workers for chunk analysis (default: {DEFAULT_MAX_PARALLEL_CHUNKS})'
    )
    
    parser.add_argument(
        '-C', '--context-lines',
        type=int,
        default=DEFAULT_CONTEXT_LINES,
        help=f'Number of lines to include before and after each relevant line for context (default: {DEFAULT_CONTEXT_LINES})'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f'Chunk size in tokens for large log processing (default: {DEFAULT_CHUNK_SIZE} tokens)'
    )
    
    parser.add_argument(
        '-P', '--preset',
        choices=list(list_presets().keys()),
        help=f'Analysis preset that combines patterns and prompts (choices: {", ".join(list_presets().keys())}). Default: {DEFAULT_PRESET}'
    )
    
    parser.add_argument(
        '--patterns-file',
        help='Path to file containing specific patterns to look for (overrides preset)'
    )
    
    parser.add_argument(
        '--prompt-file',
        help='Path to prompt intro file (overrides preset)'
    )
    
    parser.add_argument(
        '--prompt',
        help='Prompt intro text (alternative to --prompt-file)'
    )
    
    parser.add_argument(
        '-o', '--output-dir', 
        default=DEFAULT_OUTPUT_DIR,
        help='Directory to save output files (default: current directory)'
    )
    
    parser.add_argument(
        '--summary-file',
        help='Save analysis summary to specified file (optional, e.g., analysis_summary.txt)'
    )
    
    parser.add_argument(
        '-k', '--api-key',
        help='API key for the selected provider (or use environment variables: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY, AWS_BEARER_TOKEN_BEDROCK)'
    )
    
    parser.add_argument(
        '-l', '--list-providers',
        action='store_true',
        help='List available LLM providers and their status (use this without specifying a log file)'
    )
    
    parser.add_argument(
        '--filter-keywords',
        nargs='+',
        default=DEFAULT_FILTER_KEYWORDS,
        help=f'Keywords to filter relevant log lines, separate by space (default: {DEFAULT_FILTER_KEYWORDS})'
    )
    
    parser.add_argument(
        '--filter-regex',
        default=None,
        help='Custom regex pattern to filter relevant log lines (combined with --filter-keywords via OR logic)'
    )
    
    parser.add_argument(
        '--additional-context-file',
        default='',
        help='Path to file containing additional context for the analysis (e.g., known issues, solutions, project-specific information)'
    )
    
    parser.add_argument(
        '--chunk-model',
        help='Model to use for chunk analysis (if not specified, uses provider default)'
    )
    
    parser.add_argument(
        '--aggregation-model', 
        help='Model to use for aggregation analysis (if not specified, uses provider default)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        help='Timeout in seconds for API requests (default: no timeout). Useful for CI environments to prevent hanging.'
    )
    
    parser.add_argument(
        '--failure-classes',
        type=str,
        default=None,
        help=('Closed list of failure-class identifiers the model must choose from, '
              'comma-separated, or @PATH to read one per line. Each entry may carry a '
              'description as "identifier: what it means", which the model is told to '
              'match the failure against - use @PATH for those, since a description may '
              'contain commas. When omitted no classification is requested and the prompt '
              'is unchanged. Intended to be fed from whatever tool owns the enum, so the '
              'list has a single home.')
    )

    parser.add_argument(
        '--failure-class-fallback',
        type=str,
        default=None,
        help=('Which identifier from --failure-classes to record when the model returns a '
              'class outside the list or a response that cannot be parsed. Must be one of '
              'the supplied identifiers. Defaults to "unknown" when the list contains it, '
              'and is required otherwise.')
    )

    parser.add_argument(
        '-v', '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress info logs (only show warnings and errors)'
    )
    
    parser.add_argument(
        '-d', '--debug', 
        action='store_true',
        help='Enable debug files'
    )
    
    args = parser.parse_args()
    
    if args.list_providers:
        print("🤖 Available LLM Providers")
        print("=" * 30)
        
        status = get_provider_status()
        for provider, info in status.items():
            icon = "✅" if info["available"] else "❌"
            print(f"{icon} {provider:<8} - API Key: {info['api_key_preview']}")
        
        print("\nSet API keys via environment variables:")
        print("  export GOOGLE_API_KEY='your_gemini_key'        # For Gemini")
        print("  export OPENAI_API_KEY='your_openai_key'        # For OpenAI")
        print("  export ANTHROPIC_API_KEY='your_claude_key'     # For Claude")
        print("  export AWS_BEARER_TOKEN_BEDROCK='your_token'   # For AWS Bedrock")
        print("  export AWS_REGION='us-east-1'                  # For AWS Bedrock (optional)")
        return
    
    if not args.list_providers and not args.log:
        parser.error("log file is required for analysis (use --list-providers to see available providers)")
        return
    
    try:
        failure_classes, failure_class_fallback, class_config_warning = load_failure_class_config(
            args.failure_classes, args.failure_class_fallback
        )
        if class_config_warning:
            # stderr rather than the logger: logging is not configured until
            # LogAnalyzer is built, and --quiet must not hide a misconfigured enum.
            print(f"WARNING: {class_config_warning}", file=sys.stderr)

        analyzer = LogAnalyzer(
            provider=args.provider,
            api_key=args.api_key,
            output_dir=args.output_dir,
            verbose=args.verbose,
            quiet=args.quiet,
            debug=args.debug,
            max_chunks=args.max_chunks,
            max_workers=args.max_workers,
            context_lines=args.context_lines,
            chunk_size=args.chunk_size,
            filter_keywords=args.filter_keywords,
            filter_regex=args.filter_regex,
            preset=args.preset,
            patterns_file=args.patterns_file,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            additional_context_file=args.additional_context_file,
            chunk_model=args.chunk_model,
            aggregation_model=args.aggregation_model,
            timeout=args.timeout,
            failure_classes=failure_classes,
            failure_class_fallback=failure_class_fallback
        )
        
        result, has_failures = analyzer.analyze_log_file(args.log, args.tail_lines)
        
        _, markdown_path = analyzer.save_results(result)
        
        provider_info = analyzer.llm_client.get_provider_info()
        
        summary_text = format_analysis_summary(result, provider_info, markdown_path)
        
        if args.summary_file:
            # Respect output directory for summary file path
            summary_file_path = Path(args.output_dir) / args.summary_file if not Path(args.summary_file).is_absolute() else Path(args.summary_file)
            save_analysis_summary(summary_text, summary_file_path, analyzer.logger)
        
        # Only print summary if not in quiet mode
        if not args.quiet:
            print("=" * 60)
            print("LLM LOG ANALYSIS COMPLETE")
            print("=" * 60)
            print(summary_text)
        
        # Exit with appropriate code for CI
        if has_failures:
            print("\n❌ Failures detected!")
            sys.exit(1)
        else:
            print("\n✅ No significant failures detected.")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()


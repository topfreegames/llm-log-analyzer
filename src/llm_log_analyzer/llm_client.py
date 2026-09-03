"""
Flexible LLM client for log root cause analysis.
Supports multiple LLM providers using strategy pattern.
"""

import json
import logging
import re
import os
from typing import Dict, Any, Mapping, Optional, List, Sequence, Union
from dataclasses import dataclass
from llm_log_analyzer.constants import CHUNK_PROMPT_FILE, AGGREGATION_PROMPT_FILE, CHUNK_RESPONSE_FILE, AGGREGATION_RESPONSE_FILE
from llm_log_analyzer.providers import LLMProvider

ANALYSIS_FORMAT_INSTRUCTIONS = """
Provide your analysis as a structured JSON object:

```json
{
  "root_cause": "Detailed description of the primary root cause",
  "relevant_lines": ["Most relevant log lines that led to this conclusion"],
  "confidence": 0.85,
  "action_suggestion": "Specific actionable steps to fix the root cause",
  "human_summary": "Provide a concise summary (up to 6 sentences) suitable for Slack notifications or merge request comments. Focus on what went wrong and how to fix it."
}
```
**confidence** is a number between 0 and 1 that represents your confidence in the analysis - be conservative here.

IMPORTANT: 
- Use only valid JSON format with proper escaping
- Do not include comments, extra newlines, or formatting in the JSON
- Escape any quotes or special characters in strings
- Keep string values concise and on single lines on root cause and action suggestion
- For breaking lines in human summary, use `\\n`
"""

# Appended only when the caller supplies a class list, so the default prompt and
# the default output are byte-for-byte unchanged for everyone else.
FAILURE_CLASS_INSTRUCTIONS = """

Additionally, include a `"failure_class"` key whose value is EXACTLY one of the
following identifiers, chosen for the failure you just described:

{classes}

Rules for `failure_class`:
- Copy one identifier verbatim. Do not invent, abbreviate or combine them.
{match_rule}- Choose `{fallback}` when no identifier fits. A wrong identifier is worse than
  `{fallback}`, because it sends the failure to the wrong owner.
- It is independent of `confidence`: report the class you believe is right and
  let `confidence` carry your certainty.
"""

# Only used when the caller's own enum contains it. Nothing here invents a class:
# see resolve_failure_class_fallback.
DEFAULT_UNCLASSIFIED_FALLBACK = "unknown"

# A fact about the request rather than the model's opinion of it: whether the
# caller's additional-context file reached the prompt that chose the class.
# Context is known to change the class the model picks, so a consumer can require
# it before acting on one - unlike `confidence`, which is the model grading its
# own answer.
CLASSIFICATION_CONTEXT_PROVIDED = "provided"
CLASSIFICATION_CONTEXT_ABSENT = "absent"

# What a caller may hand us: bare identifiers, or identifier -> description.
FailureClassSpec = Union[Sequence[str], Mapping[str, str]]


def normalize_failure_classes(failure_classes: Optional[FailureClassSpec] = None) -> Dict[str, str]:
    """Coerce either accepted shape into an ordered identifier -> description map.

    A bare sequence yields empty descriptions, which is still a usable enum - the
    identifier then has to carry its own meaning.
    """
    if not failure_classes:
        return {}
    items = failure_classes.items() if isinstance(failure_classes, Mapping) else ((c, "") for c in failure_classes)
    normalized: Dict[str, str] = {}
    for identifier, description in items:
        key = str(identifier or "").strip()
        if key:
            normalized[key] = str(description or "").strip()
    return normalized


def resolve_failure_class_fallback(failure_classes: Optional[FailureClassSpec] = None,
                                   explicit: Optional[str] = None) -> str:
    """The identifier to record when the model's answer is unusable.

    It has to be a member of the caller's own enum. Hardcoding one here would let
    this module emit a value outside the closed list it promises to honour, and
    would break silently the day the owner of the enum renamed that class.
    """
    classes = normalize_failure_classes(failure_classes)
    if not classes:
        return ""
    if explicit and explicit.strip():
        candidate = explicit.strip()
        if candidate not in classes:
            raise ValueError(
                f"failure-class fallback {candidate!r} is not one of the supplied classes: "
                f"{', '.join(classes)}"
            )
        return candidate
    if DEFAULT_UNCLASSIFIED_FALLBACK in classes:
        return DEFAULT_UNCLASSIFIED_FALLBACK
    raise ValueError(
        f"{DEFAULT_UNCLASSIFIED_FALLBACK!r} is not among the supplied classes, so there is "
        f"no default to fall back on; name one explicitly out of: {', '.join(classes)}"
    )


def build_analysis_format_instructions(failure_classes: Optional[FailureClassSpec] = None,
                                       fallback: Optional[str] = None) -> str:
    """Schema block for the aggregation prompt, plus the class enum when supplied."""
    classes = normalize_failure_classes(failure_classes)
    if not classes:
        return ANALYSIS_FORMAT_INSTRUCTIONS
    fallback = fallback or resolve_failure_class_fallback(classes)
    listed = "\n".join(
        f"  - {identifier}: {description}" if description else f"  - {identifier}"
        for identifier, description in classes.items()
    )
    # With descriptions, point the model at them: matching a described signature is
    # a far more reliable ask than inferring intent from an identifier alone.
    match_rule = (
        "- Match the failure against the descriptions above and choose the one it fits.\n"
        "  Where a description names a log signature, that signature is what decides.\n"
        if any(classes.values()) else
        "- Choose the identifier that best fits the failure you described.\n"
    )
    return ANALYSIS_FORMAT_INSTRUCTIONS + FAILURE_CLASS_INSTRUCTIONS.format(
        classes=listed, match_rule=match_rule, fallback=fallback
    )

CHUNK_FORMAT_INSTRUCTIONS = """
Provide a concise summary (up to 10 sentences) of the most important issues found in this chunk. Include the main relevant log lines that led to this conclusion (up to 5 lines).

If no issues are found, state "No significant issues detected in this chunk."""

CHUNK_ANALYSIS_INSTRUCTION = "Analyze the following log chunk and identify any potential signs that could lead to failures."

AGGREGATION_ANALYSIS_INSTRUCTION = "Based on the context provided, determine the root cause of the failure."

@dataclass
class AnalysisResult:
    """Structured result from log analysis."""
    root_cause: str
    relevant_lines: List[str]
    confidence: float
    action_suggestion: str
    human_summary: str
    failure_class: str = ""
    classification_context: str = ""

class LLMAnalyzer:
    """Handles analysis logic and prompting using an LLM provider."""
    
    def __init__(self, provider: LLMProvider, debug_output_dir: Optional[str] = None,
                 failure_classes: Optional[FailureClassSpec] = None,
                 failure_class_fallback: Optional[str] = None):
        self.provider = provider
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.debug_output_dir = debug_output_dir
        self.failure_classes = normalize_failure_classes(failure_classes)
        self.failure_class_fallback = resolve_failure_class_fallback(
            self.failure_classes, failure_class_fallback
        )
    
    def _save_debug_prompt(self, prompt: str, filename: str) -> None:
        """Save the final formatted prompt to a debug file."""
        if self.debug_output_dir is None:
            return
        
        try:
            debug_path = os.path.join(self.debug_output_dir, filename)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            self.logger.debug(f"Saved debug prompt to {debug_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save debug prompt to {filename}: {e}")
    
    def _save_debug_response(self, response: str, filename: str) -> None:
        """Save the raw model response to a debug file."""
        if self.debug_output_dir is None:
            return
        
        try:
            debug_path = os.path.join(self.debug_output_dir, filename)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(response)
            self.logger.debug(f"Saved debug response to {debug_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save debug response to {filename}: {e}")
    
    def analyze_chunk(self, log_chunk: str, chunk_number: int, model: Optional[str] = None, custom_prompt: Optional[str] = None, additional_context: str = '') -> str:
        """Analyze a single chunk of log data."""
        if model is None:
            model = self.provider.get_default_chunk_model()
        
        prompt_intro = custom_prompt
        
        formatted_prompt = prompt_intro + "\n\n" + CHUNK_ANALYSIS_INSTRUCTION
        formatted_prompt += "\n\n---\n\n" + log_chunk
        formatted_prompt += "\n\n---\n\n" + CHUNK_FORMAT_INSTRUCTIONS
        
        if additional_context.strip():
            formatted_prompt += f"\n\n---\n\nAdditional Context:\n{additional_context}"
        
        self._save_debug_prompt(formatted_prompt, CHUNK_PROMPT_FILE.format(chunk_number=chunk_number))
        
        response = self.provider.make_request(model, formatted_prompt)
        self._save_debug_response(response, CHUNK_RESPONSE_FILE.format(chunk_number=chunk_number))
        
        return response
    
    def aggregate_analysis(self, chunk_summaries: List[str], model: Optional[str] = None, custom_prompt: Optional[str] = None, additional_context: str = '') -> AnalysisResult:
        """Aggregate chunk summaries and determine the root cause."""
        if model is None:
            model = self.provider.get_default_aggregation_model()
        
        prompt_intro = custom_prompt
                
        chunk_summaries_text = chr(10).join([f"Chunk {i+1}:\n{summary}\n---\n" for i, summary in enumerate(chunk_summaries)])

        chunk_header_text = "Log chunk summaries:\n\n" if len(chunk_summaries) > 1 else ""
        
        formatted_prompt = prompt_intro + "\n\n" + AGGREGATION_ANALYSIS_INSTRUCTION
        formatted_prompt += "\n\n---\n\n" + chunk_header_text + chunk_summaries_text
        formatted_prompt += build_analysis_format_instructions(
            self.failure_classes, self.failure_class_fallback
        )
        
        has_context = bool(additional_context.strip())
        if has_context:
            formatted_prompt += f"\n\n---\n\nAdditional Context:\n{additional_context}"
        
        # Recorded from the prompt we actually sent, not from what the model says
        # about it. Only meaningful when a class was asked for.
        classification_context = ""
        if self.failure_classes:
            classification_context = (
                CLASSIFICATION_CONTEXT_PROVIDED if has_context else CLASSIFICATION_CONTEXT_ABSENT
            )
        
        self._save_debug_prompt(formatted_prompt, AGGREGATION_PROMPT_FILE)
        
        response = self.provider.make_request(model, formatted_prompt)
        self._save_debug_response(response, AGGREGATION_RESPONSE_FILE)
        
        return self._parse_aggregation_response(response, classification_context)
    
    def _validate_failure_class(self, value: Any) -> str:
        """Coerce the model's class to one we asked for; anything else is the fallback."""
        if not self.failure_classes:
            return ""
        candidate = str(value or "").strip()
        if candidate in self.failure_classes:
            return candidate
        if candidate:
            self.logger.warning(
                f"Model returned failure_class {candidate!r}, which is not in the "
                f"supplied enum; recording {self.failure_class_fallback!r} instead"
            )
        return self.failure_class_fallback

    def _parse_aggregation_response(self, response: str, classification_context: str = "") -> AnalysisResult:
        """Parse the aggregation response to extract structured data."""
        try:
            json_str = self._sanitize_json_string(response.strip())
            analysis_data = json.loads(json_str)
            
            return AnalysisResult(
                root_cause=analysis_data.get("root_cause", "Unknown root cause"),
                relevant_lines=analysis_data.get("relevant_lines", []),
                confidence=float(analysis_data.get("confidence", 0.5)),
                action_suggestion=analysis_data.get("action_suggestion", "No specific action suggested"),
                human_summary=analysis_data.get("human_summary", "No human summary was found in the response"),
                failure_class=self._validate_failure_class(analysis_data.get("failure_class")),
                classification_context=classification_context,
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"Raw response excerpt: {response[:500]}...")
            self.logger.debug(f"Extracted JSON block: {repr(json_str)}")
            self.logger.debug(f"Full response: {repr(response)}")
            
            return self._fallback_parse_response(response, str(e), classification_context)
            
        except Exception as e:
            self.logger.error(f"Failed to parse aggregation response: {e}")
            self.logger.debug(f"Response was: {response[:1000]}...")
            
            return AnalysisResult(
                root_cause="Failed to parse analysis results",
                relevant_lines=[],
                confidence=0.0,
                action_suggestion="Review logs manually",
                human_summary=f"Analysis failed due to parsing error: {str(e)}",
                failure_class=self.failure_class_fallback,
                classification_context=classification_context,
            )
    
    def _sanitize_json_string(self, json_str: str) -> str:
        """Clean and sanitize JSON string to remove problematic characters."""
        import re

        json_markdown = '```json'
        json_start = json_str.find(json_markdown)
        if json_start != -1:
            json_end = json_str.find('```', json_start + len(json_markdown))
            if json_end != -1:
                json_str = json_str[json_start + len(json_markdown):json_end].strip()
        
        try:
            def fix_string_value(match):
                content = match.group(1)
                content = content.replace('\n', '\\n')
                content = content.replace('\r', '\\r') 
                content = content.replace('\t', '\\t')
                content = content.replace('\b', '\\b')
                content = content.replace('\f', '\\f')
                content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
                return f'"{content}"'
            
            json_str = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', fix_string_value, json_str)
            
            lines = []
            for line in json_str.split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    lines.append(line)
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.warning(f"Advanced JSON sanitization failed: {e}, using basic cleanup")
            
            # Fallback: More aggressive cleaning
            # Remove obvious control characters that shouldn't be in JSON structure
            json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', json_str)
            
            lines = []
            for line in json_str.split('\n'):
                line = line.strip()
                if line:
                    lines.append(line)
            
            return '\n'.join(lines)
    
    def _fallback_parse_response(self, response: str, error_msg: str,
                                 classification_context: str = "") -> AnalysisResult:
        """Fallback parsing when JSON fails - extract info using regex."""
        self.logger.info("Attempting fallback parsing using regex extraction")
        
        root_cause = "Unable to parse root cause from response"
        confidence = 0
        action_suggestion = "Review the raw LLM response manually"
        failure_class = self.failure_class_fallback
        
        try:
            root_cause_match = re.search(r'"root_cause":\s*"([^"]*)"', response)
            if root_cause_match:
                root_cause = root_cause_match.group(1)
            
            confidence_match = re.search(r'"confidence":\s*([0-9.]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
            
            action_match = re.search(r'"action_suggestion":\s*"([^"]*)"', response)
            if action_match:
                action_suggestion = action_match.group(1)
            
            # Malformed JSON elsewhere is no reason to discard a class the model did
            # emit; validation still gates what we accept.
            if self.failure_classes:
                class_match = re.search(r'"failure_class":\s*"([^"]*)"', response)
                if class_match:
                    failure_class = self._validate_failure_class(class_match.group(1))
                
            summary_start = response.find('HUMAN SUMMARY:')
            if summary_start != -1:
                human_summary = response[summary_start + 14:].strip()
            else:
                human_summary = f"JSON parsing failed: {error_msg}. Extracted some info using fallback method."
                
        except Exception as e:
            self.logger.error(f"Fallback parsing also failed: {e}")
            human_summary = f"Both primary and fallback parsing failed. Original error: {error_msg}"
        
        return AnalysisResult(
            root_cause=root_cause,
            relevant_lines=[],
            confidence=confidence,
            action_suggestion=action_suggestion,
            human_summary=human_summary,
            failure_class=failure_class,
            classification_context=classification_context,
        )

class LLMClient:
    """Main LLM client that uses different providers and analyzers."""
    
    def __init__(self, provider, debug_output_dir: str = ".", chunk_model: Optional[str] = None, aggregation_model: Optional[str] = None, failure_classes: Optional[FailureClassSpec] = None, failure_class_fallback: Optional[str] = None):
        """
        Initialize the LLM client with a specific provider.
        
        Args:
            provider: LLM provider to use
            debug_output_dir: Directory for debug output files
            chunk_model: Model to use for chunk analysis (if not provided, uses provider default)
            aggregation_model: Model to use for aggregation analysis (if not provided, uses provider default)
            failure_classes: Closed set of failure-class identifiers the model must choose
                from, either bare identifiers or identifier -> description. When omitted,
                no classification is requested and the prompt is unchanged.
            failure_class_fallback: Which of those identifiers to record when the model's
                answer is unusable. Defaults to "unknown" when the set contains it.
        """
        self.provider_type = type(provider).__name__
        self.provider = provider
        self.logger = logging.getLogger(__name__)
        self.chunk_model = chunk_model or self.provider.get_default_chunk_model()
        self.aggregation_model = aggregation_model or self.provider.get_default_aggregation_model()
        self.analyzer = LLMAnalyzer(self.provider, debug_output_dir, failure_classes=failure_classes,
                                    failure_class_fallback=failure_class_fallback)

        self.logger.info(f"Initialized LLM client with provider {self.provider_type}, aggregation model {self.aggregation_model}, chunk model {self.chunk_model}")
    
    def analyze_chunk(self, log_chunk: str, chunk_number: int, model: Optional[str] = None, custom_prompt: Optional[str] = None, additional_context: str = '') -> str:
        """Analyze a chunk using the current analyzer."""
        return self.analyzer.analyze_chunk(log_chunk, chunk_number, self.chunk_model, custom_prompt, additional_context)
    
    def aggregate_analysis(self, chunk_summaries: List[str], model: Optional[str] = None, custom_prompt: Optional[str] = None, additional_context: str = '') -> AnalysisResult:
        """Aggregate analysis using the current analyzer."""
        return self.analyzer.aggregate_analysis(chunk_summaries, self.aggregation_model, custom_prompt, additional_context)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            "provider": self.provider_type,
            "default_chunk_model": self.chunk_model,
            "default_aggregation_model": self.aggregation_model
        }

# LLM Log Analyzer

[![PyPI version](https://badge.fury.io/py/llm-log-analyzer.svg)](https://badge.fury.io/py/llm-log-analyzer)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A log analyzer for CI/CD pipelines (Jenkins, GitLab CI, GitHub Actions) that leverages Large Language Models to automatically detect and explain the root cause of failures, suggesting possible fixes.

This project originated from the difficulty of diagnosing issues in Unity build logs, which are notoriously verbose and hard to navigate — especially for less experienced developers. While optimized for Unity by default, the analyzer is fully configurable and can be adapted to any kind of build or runtime logs.

Feel free to contribute with new presets or refinements.

## Features

- **Efficient Processing**: Handles very large logs (>50MB) using streaming techniques
- **Flexible AI Models**: Support for multiple LLM models (Claude, Gemini, OpenAI models)
- **Structured Output**: Provides both human-friendly summaries and machine-readable JSON
- **CI-Ready**: Designed for seamless integration with Jenkins, GitLab CI, and other platforms
- **Configurable**: Customizable error patterns and prompts for different log types
- **Smart Filtering**: Extracts only relevant error information to optimize LLM results
- **Unity-Optimized**: Includes Unity-specific and generic patterns by default

## Installation

### Using pip (Recommended)

```bash
pip install llm-log-analyzer
```

### From GitHub (Development)

```bash
pip install git+https://github.com/renatoaf/llm-log-analyzer
# or pip3 install git+ssh://git@github.com/renatoaf/llm-log-analyzer
```

### Cloning the repo

1. Clone this repository:
```bash
git clone https://github.com/renatoaf/llm-log-analyzer
cd llm-log-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

Set up API keys for your preferred LLM provider:

**For Gemini:**
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
```

**For OpenAI:**
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

**For Claude:**
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

or

```bash
export AWS_BEARER_TOKEN_BEDROCK="your_aws_bearer_token"
export AWS_REGION="your_aws_region" # defaults to us-east-1
```

For using Claude through AWS Bedrock.

**Note**: The analyzer will default to Gemini. You can get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Usage

The usages refer to the package being installed via pip. If you cloned the repo, replace `llm-log-analyzer` by `python3 analyze_log.py` (inside `src/llm_log_analyzer`).

### Basic Usage

```bash
# Check available LLM providers and their status
llm-log-analyzer --list-providers

# Basic analysis (uses generic preset by default)
llm-log-analyzer application.log

# Use Unity preset for Unity build logs  
llm-log-analyzer build.log -P unity

# Specify LLM provider with presets
llm-log-analyzer server.log -P generic --provider openai
llm-log-analyzer build.log -P unity --provider gemini
llm-log-analyzer build.log -P unity --provider claude

# Use custom API key with presets
llm-log-analyzer system.log -P generic --provider openai --api-key your_key
llm-log-analyzer build.log -P unity --provider claude --api-key your_key

# Advanced options with presets
llm-log-analyzer app.log -P generic --tail-lines 5000 --output-dir results/ --debug

# Manual file specification
llm-log-analyzer server.log --patterns-file patterns/generic.txt --prompt-file prompts/generic.txt
```

### Command Line Options

Run:
```
llm-log-analyzer -h
```

For detailed options and usage examples.

### Analysis Presets

The analyzer now supports **presets** that combine pattern and prompt files for common use cases. This makes it easier to switch between different types of log analysis.

#### Available Presets

| Preset | Description | Pattern File | Prompt File |
|--------|-------------|-------------|-------------|
| `generic` | General purpose analysis for any application logs | `patterns/generic.txt` | `prompts/generic.txt` |
| `unity` | Unity game engine build logs with game-specific patterns | `patterns/unity.txt` | `prompts/unity.txt` |
| `android` | Native Android builds analysis | `patterns/android.txt` | `prompts/android.txt` |
| `ios` | Native iOS builds analysis | `patterns/ios.txt` | `prompts/ios.txt` |
| `flutter` | Flutter cross-platform builds analysis | `patterns/flutter.txt` | `prompts/flutter.txt` |

#### Using Presets

```bash
# Use preset (recommended approach)
llm-log-analyzer build.log -P unity
llm-log-analyzer server.log -P generic
llm-log-analyzer gradle-build.log -P android
llm-log-analyzer xcode-build.log -P ios
llm-log-analyzer flutter-build.log -P flutter

# Override individual files while using a preset
llm-log-analyzer build.log -P unity --prompt "Focus on memory leaks in Unity builds"
llm-log-analyzer build.log -P generic --patterns-file my-custom-patterns.txt

# Manual file specification
llm-log-analyzer build.log --patterns-file patterns/unity.txt --prompt-file prompts/unity.txt
```

**Default Behavior:** If no preset is specified, the analyzer defaults to the `generic` preset, making it suitable for any type of log analysis out of the box. Beware that this wasn't optimized yet, so results may feel generic.

### Output Files

The analyzer generates two output files:

1. **`analysis.json`** - Machine-readable structured analysis:
```json
{
  "root_cause": "Script compilation failed due to a missing definition for 'Armadillo' in the 'CharacterEnum' enum. This suggests a new character was added without updating the enum accordingly.",
  "relevant_lines": [
    "Assets/Scripts/Gameplay/Controllers/Player/CharacterSoundControllerFactory.cs(148,66): error CS0117: 'CharacterEnum' does not contain a definition for 'Armadillo'",
    "Error building Player because scripts have compile errors in the editor"
  ],
  "confidence": 0.95,
  "action_suggestion": "Add a definition for the 'Armadillo' character to the 'CharacterEnum' enum in the CharacterSoundControllerFactory.cs file. Ensure the enum is updated whenever new characters are introduced.",
  "human_summary": "The Unity build failed due to a script compilation error. The error message indicates that `CharacterEnum` is missing a definition for 'Armadillo'.  This likely happened because a new character, 'Armadillo', was added without updating the corresponding enum. To fix this, add a new entry for 'Armadillo' to the `CharacterEnum` enum in the `CharacterSoundControllerFactory.cs` file.",
  "failure_class": "",
  "classification_context": ""
}
```

Both fields are empty unless `--failure-classes` is supplied; see
[Classifying failures](#classifying-failures).

### Classifying failures

`--failure-classes` asks the model to place the failure into a **closed list** of
identifiers you supply, and records its choice in `failure_class`:

```bash
# inline, bare identifiers
llm-log-analyzer build.log --failure-classes licensing_down,project_code,unknown

# or from a file, one entry per line (`#` comments allowed)
llm-log-analyzer build.log --failure-classes @enum.txt
```

Each entry may carry a description, which is what the model is told to match the
failure against:

```
# enum.txt
licensing_down: The shared licence server could not issue a licence - exit 198.
project_code: Compilation failure with diagnostics present in the log.
unknown: Nothing in the log corroborates any class above.
unknown_no_signal: The build failed without recording why.
```

Descriptions are worth writing. An identifier alone leaves the model inferring what
you meant, and near-neighbours — `unknown` against `unknown_no_signal`, one
licensing failure mode against another — are not distinguishable from the name.
Use `@PATH` for described enums, since a description may contain commas.

The analyzer deliberately does **not** ship an enum of its own. Whatever consumes
the classification — a metrics exporter, a dashboard, an alerting rule — owns the
list and passes it in, so there is exactly one definition to keep in step with the
queries built on it.

Guarantees worth relying on:

- **Opt-in.** Without the flag the prompt and the output are unchanged, and both
  fields are `""`.
- **Closed.** A value outside your list is replaced with the fallback class and
  logged as a warning. The model cannot widen the enum by inventing an identifier.
- **Always present when asked.** Every path — a value outside the list, a missing
  key, an unparseable response — yields a class from *your* list rather than an
  empty or absent field, so a consumer can distinguish "the analyzer ran and could
  not classify" from "the analyzer never ran".
- **The fallback is yours too.** It defaults to `unknown` when your list contains
  that identifier, and `--failure-class-fallback` names a different one.
- **Never fatal.** A missing enum file, an empty list, or a fallback that resolves
  to nothing warns on stderr and disables classification for that run - it does not
  fail the analysis. `failure_class` is then `""`, exactly as if no class had been
  asked for, and the log analysis a developer reads is unaffected. Classification
  is an addition to the analysis, not a precondition for it.

  What it will never do is pick an arbitrary member of your list as the fallback.
  Recording a real class when the model meant "no idea" misattributes the failure,
  which is worse than recording nothing.

#### Was the class chosen with context?

`classification_context` records whether `--additional-context-file` reached the
prompt that chose the class: `provided` or `absent`.

It is a fact about the request, not the model's opinion of it. `confidence` is the
model grading its own answer, and does not track correctness — the same 0.99 will
sit on a wrong class and on its corrected replacement. Whether the model had your
known-errors context when it classified is checkable, and it measurably changes the
class it picks, so a consumer can require context before acting on a class.

2. **`analysis.md`** - Human-friendly Markdown summary for Slack/MR comments:
```markdown
# Build Analysis Report

## Summary
The Unity build failed due to a script compilation error. The error message indicates that `CharacterEnum` does not contain a definition for `Armadillo`.  This likely means a new character was added without updating the enum. To fix this, add `Armadillo` to the `CharacterEnum` or correct the reference in `CharacterSoundControllerFactory.cs` if a different character was intended.

## Analysis Configuration

**Provider:** GeminiProvider

**Models Used:** 
- Chunk Analysis: gemini-flash-latest
- Final Aggregation: gemini-pro-latest

**Processing Settings:**
- Max Chunks: 20
- Max Workers: 4

## Analysis Details

**Root Cause:** Script compilation error due to missing definition for 'Armadillo' in 'CharacterEnum'

**Confidence:** 🟢 95.0%

## Action Items
Add the 'Armadillo' definition to the 'CharacterEnum' or correct the reference in 'CharacterSoundControllerFactory.cs' if it's intended to be a different character.

## Relevant Log Lines
Assets/Scripts/Gameplay/Controllers/Player/CharacterSoundControllerFactory.cs(148,66): error CS0117: 'CharacterEnum' does not contain a definition for 'Armadillo'

Error building Player because scripts have compile errors in the editor
```

### Exit Codes

- `0`: No significant failures detected in log
- `1`: Failures found and analyzed
- `2`: Analysis failed (e.g., file not found, API errors)

## LLM Provider Comparison

The analyzer supports multiple LLM providers to fit different needs and budgets:

### Claude (Anthropic)
- **Cost**: Pay-per-use pricing, very expensive 
- **Speed**: Fast, but more expensive per request
- **Accuracy**: Excellent, especially for complex log analysis and suggestions for solution
- **Setup**: Requires paid Anthropic account

**Best for**: Complex logs, when you need solution accuracy

### GPT (OpenAI)
- **Cost**: Pay-per-use pricing with competitive rates
- **Speed**: Fast with good throughput
- **Accuracy**: Good for most log analysis tasks
- **Setup**: Requires OpenAI API key from [OpenAI Platform](https://platform.openai.com)

**Best for**: Balanced cost and performance

### Gemini (Google)
- **Cost**: Free tier with generous limits
- **Speed**: Very fast (especially Gemini 2.5 Flash)
- **Accuracy**: Good for most log analysis tasks
- **Setup**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

**Best for**: CI environments, frequent analysis

Gemini works exceptionally well for Unity builds use case. Given its strong performance and competitive pricing, it is currently the recommended option for running this analyzer in CI environments.

## CI Integration

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        GOOGLE_API_KEY = credentials('google-api-key')
        OPENAI_API_KEY = credentials('openai-api-key') # Alternative
        ANTHROPIC_API_KEY = credentials('anthropic-api-key') # Alternative
    }
    
    stages {
        stage('Application Build') {
            steps {
                sh 'make build > build.log 2>&1'
            }
            post {
                failure {
                    // Only run analyzer when build fails
                    echo "Build failed, analyzing logs..."
                    sh '''
                        llm-log-analyzer -P unity --output-dir ./analysis/ build.log
                        
                        # Check if analysis found failures
                        if [ $? -eq 1 ]; then
                            echo "Build failures detected and analyzed"
                        fi
                    '''
                    
                    // Archive analysis results
                    archiveArtifacts artifacts: 'analysis/*', allowEmptyArchive: true
                    
                    // Publish results to Slack/Teams
                    script {
                        if (fileExists('analysis/analysis.md')) {
                            def analysis = readFile('analysis/analysis.md')
                            slackSend(
                                channel: '#builds',
                                color: 'danger',
                                message: "🚨 Build Failed - Analysis Results:\n```${analysis}```"
                            )
                        }
                    }
                }
                always {
                    // Archive build log for reference
                    archiveArtifacts artifacts: 'build.log', allowEmptyArchive: true
                    archiveArtifacts artifacts: 'analysis.md', allowEmptyArchive: true
                }
            }
        }
    }
}
```

### GitLab CI

```yaml
# .gitlab-ci.yml
variables:
  GOOGLE_API_KEY: $GOOGLE_API_KEY
  OPENAI_API_KEY: $OPENAI_API_KEY # Alternative
  ANTHROPIC_API_KEY: $ANTHROPIC_API_KEY # Alternative

stages:
  - build
  - analyze

build_app:
  stage: build
  script:
    - make build > build.log 2>&1
  artifacts:
    when: always
    paths:
      - build.log
    expire_in: 1 day

log_analysis:
  stage: analyze
  dependencies:
    - build_app
  script:
    ... install llm-log-analyzer ...
    - llm-log-analyzer build.log --output-dir analysis/ -P unity
    - |
      if [ -f analysis/analysis.md ]; then
        echo "## 🚨 Build Failed - Log Analysis Results" >> analysis_comment.md
        cat analysis/analysis.md >> analysis_comment.md
        
        # Optional: Post to merge request if available
        if [ -n "$CI_MERGE_REQUEST_IID" ]; then
          curl -X POST \
            -H "PRIVATE-TOKEN: $GITLAB_TOKEN" \
            -F "body=@analysis_comment.md" \
            "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
        fi
      fi
  artifacts:
    when: always
    paths:
      - analysis/
    expire_in: 1 week
  when: on_failure
```

## Customization

The analyzer is highly configurable and can be adapted for different log types, environments, and specific project needs.

### Configuration for Different Log Types

#### Generic Application Logs (Default)
By default, the analyzer uses the generic preset suitable for any application logs:

```bash
llm-log-analyzer app.log  # Uses generic preset by default
llm-log-analyzer app.log -P generic  # Explicitly specify generic preset
```

#### Unity Build Logs
For Unity build logs, use the Unity preset with game-specific patterns and prompts:

```bash
llm-log-analyzer build.log -P unity  # Uses Unity-specific patterns and prompts
```

#### Custom Log Types
Create your own patterns and prompts for specific applications:

1. **Create custom patterns file** (`custom_patterns.txt`):
```text
# Custom error patterns for MyApp
MyApp ERROR:
FATAL:
Application crashed
Database connection failed
```

2. **Create custom prompt file** (`custom_prompt.txt`):
```text
You are an iOS specialist investigating errors in a Xcode build log...
```

3. **Use custom configuration**:
```bash
# Start with a preset and override specific parts
llm-log-analyzer build.log -P generic --prompt "You are a DevOps expert analyzing Docker container logs. Look for ..."

llm-log-analyzer xcode.log -P generic --prompt "You are an iOS Developer investigating a crash..."

# Use completely custom files
llm-log-analyzer myapp.log --patterns-file custom_patterns.txt --prompt-file custom_prompt.txt

# Mix preset patterns with custom prompt
llm-log-analyzer build.log -P unity --prompt "Focus specifically on IL2CPP compilation issues..."
```

### Additional Context for Known Issues

You can provide additional context via a file to help the analyzer understand known issues and solutions specific to your project or CI environment. This is particularly useful for:

- Documenting recurring build failures and their solutions
- Providing project-specific troubleshooting information  
- Adding context about infrastructure-specific issues
- Including known workarounds for common problems

**Creating a Context File:**

Create a text file (e.g., `context.txt`) with your known issues and solutions, example:

```text
# Unity Build Context
Known Issues:
- Unity 2021.3.5f1 has memory issues on Jenkins agents with <8GB RAM
- Android builds fail on Gradle 7.x due to dependency conflicts with com.google.firebase
- Network timeouts during package resolution in corporate networks

Common Solutions:
- Memory issues: Use agents with more memory or reduce build parallelism with --max-workers 2
- Gradle conflicts: Use Gradle 6.8 or update Firebase to v31.0.0+
- Unity license activation fails on new runners: requires manual activation or check license server connectivity

Build Environment Notes:
- Using GitLab CI with Docker containers
- Jenkins agents have variable memory (4-16GB)
- Corporate firewall may block some package repositories
...
```

**Usage:**

```bash
# Basic usage with context file
llm-log-analyzer build.log -P unity --additional-context-file context.txt

# Different context files for different environments
llm-log-analyzer unity-build.log -P unity --additional-context-file contexts/unity-context.txt
llm-log-analyzer server.log --additional-context-file contexts/server-context.txt

# Combine with other options
llm-log-analyzer build.log --additional-context-file team-knowledge.txt --verbose --debug --output-dir results/
```

**Best Practices for Context Files:**

- Keep context files versioned with your project
- Update context as you discover new issues and solutions
- Use clear headers and bullet points for readability
- Include environment-specific details (OS, versions, infrastructure)
- Document both problems AND their solutions

### Model Selection

Each provider uses optimized models for different tasks:

**OpenAI:**
- Chunk analysis: `gpt-5-mini`
- Final aggregation: `gpt-5-mini`

**Gemini:** (floating aliases — always track the latest GA models)
- Chunk analysis: `gemini-flash-latest`
- Final aggregation: `gemini-pro-latest`

**Claude:** (directly via Anthropic API)
- Chunk analysis: `claude-haiku-4-5`
- Final aggregation: `claude-sonnet-5`

**Claude via AWS Bedrock:**
- Chunk analysis: `anthropic.claude-haiku-4-5-20251001-v1:0`
- Final aggregation: `anthropic.claude-sonnet-4-5-20250929-v1:0`

You can customize these by using the parameters `--aggregation-model` and `--chunk-model`.

## Troubleshooting

### Common Issues

**API Rate Limits**
- The analyzer includes automatic retry with exponential backoff
- Consider using other models for chunk analysis to reduce costs
- Adjust `chunk-size` to reduce API calls
- Limit the number of chunks with `max-chunks`
- Fine tune your filters

**Memory Issues with Large Logs**
- The analyzer uses streaming processing to avoid loading entire logs
- Reduce tail-lines if needed
- Fine tune your filters

**Poor Analysis Quality**
- Ensure your filtered log contains actual error information
- Fine tune your error patterns to reduce noise
- Try increasing tail-lines to capture more context
- Run with debug option and check that error keywords are being detected properly by analyzing the `filtered_log.txt`
- Run with debug option and analyze the final output prompts

**Network/Connectivity**
- Set up proper proxy configuration if behind corporate firewall
- Verify API keys are correctly set
- Check if your network allows HTTPS connections to Anthropic/Gemini API

**JSON Parsing Errors (Gemini)**
- Error: `Invalid control character at: line X column Y`
- **Cause**: Gemini sometimes includes unescaped control characters in JSON responses
- **Solution**: The analyzer now automatically sanitizes JSON responses
- **Workaround**: Use a different provider if Gemini consistently fails: `--list-providers`

### Debug Mode

Enable debug mode (`-d`) and/or verbose logging (`-v`) to diagnose issues:

```bash
llm-log-analyzer -P unity Editor.log --debug
```

This will output debug files (filtered logs, prompt and model responses).

You can additionally run in verbose mode to show API request/response information and details about execution.

```bash
llm-log-analyzer -P unity Editor.log --verbose
```

## Optimization

- **Aggressive Filtering** – Irrelevant noise is stripped out automatically, keeping only error-related lines and context. This minimizes token usage while preserving diagnostic value.

- **Chunked Processing** – If the filtered log is still too large, it is split into smaller chunks. Each chunk is analyzed individually, and results are aggregated into a single root cause report.

- **Tail Limit** – Analysis prioritizes the most recent section of the log, where failures are most likely to occur, reducing cost and improving accuracy.
## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details.


---

Made with ❤️ for Unity developers tired of manually debugging build failures.

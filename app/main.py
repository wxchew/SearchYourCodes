#!/usr/bin/env python3
"""
SearchYourCodes - Intelligent Code Search and Discovery Platform

Flask web application providing an intelligent code search interface
using multiple search methods: keyword matching, UniXcoder code structure
analysis, SBERT semantic search, and RRF-fused results.
"""

import os
import sys
import signal
import atexit
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, render_template, request, jsonify
from markupsafe import escape as html_escape

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from core.search import compare_models, fuse_results
except ImportError as e:
    print(f"Error importing search modules: {e}")
    print("Make sure the core modules are available")
    sys.exit(1)

app = Flask(__name__, template_folder='templates')

# Configuration
try:
    from core.config import get_app_config, load_config
    app_config = get_app_config()
    full_config = load_config()
    app.config['MAX_CONTENT_LENGTH'] = app_config.get('max_content_length', 32 * 1024 * 1024)
    DEFAULT_MAX_RESULTS = 10
    MAX_CODE_DISPLAY_LINES = app_config.get('max_code_display_lines', 50)
    MAX_CODE_HEIGHT = app_config.get('max_code_height', 600)
    DEBUG_ROUTES_ENABLED = full_config.get('dev', {}).get('enable_debug_routes', False)
except ImportError:
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
    DEFAULT_MAX_RESULTS = 10
    MAX_CODE_DISPLAY_LINES = 50
    MAX_CODE_HEIGHT = 600
    DEBUG_ROUTES_ENABLED = False


def cleanup_resources():
    """Clean up resources on shutdown"""
    print("\nCleaning up resources...")
    gc.collect()


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    print("\nShutting down SearchYourCodes gracefully...")
    cleanup_resources()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_resources)


@app.route('/')
def index():
    """Main search interface"""
    return render_template('index.html')


@app.route('/test')
def test():
    """Test endpoint to verify template updates"""
    return "<h1>Template Test - SearchYourCodes is working!</h1>"


@app.route('/open-file')
def open_file():
    """Display file content in browser with line highlighting"""
    file_path = request.args.get('file', '').strip()
    line = request.args.get('line', '1')
    line_end = request.args.get('line_end', line)

    if not file_path:
        return "No file specified", 400

    try:
        file_path = _resolve_file_path(file_path)
        start_line, end_line = _parse_line_numbers(line, line_end)

        # Security: resolve symlinks and ensure file is within project directory
        project_root = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
        real_path = os.path.realpath(file_path)
        if not real_path.startswith(project_root + os.sep) and real_path != project_root:
            return "Access denied: File outside project directory", 403

        if not os.path.exists(real_path):
            return f"File not found: {file_path}", 404

        return _generate_file_viewer_html(real_path, start_line, end_line)

    except ValueError as e:
        return f"Invalid parameters: {str(html_escape(str(e)))}", 400
    except Exception as e:
        return f"Error reading file: {str(html_escape(str(e)))}", 500


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests with optional filters and pagination"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        query = data.get('query', '').strip()
        max_results = data.get('max_results', DEFAULT_MAX_RESULTS)
        page = data.get('page', 1)
        page_size = data.get('page_size', max_results)

        # Validate input
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        if len(query) > 500:
            return jsonify({'error': 'Query too long (max 500 characters)'}), 400
        if not isinstance(max_results, int) or max_results < 1 or max_results > 100:
            max_results = DEFAULT_MAX_RESULTS

        # Build metadata filter from request
        filter_metadata = _build_metadata_filter(data)

        # Fetch enough results for pagination
        fetch_k = max(max_results, page * page_size)

        # Perform all three searches
        keyword_results, unixcoder_results, sbert_results = compare_models(
            query, k=fetch_k, filter_metadata=filter_metadata
        )

        # Generate fused results
        fused = fuse_results(keyword_results, unixcoder_results, sbert_results,
                             max_results=fetch_k)

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        results = {
            'query': query,
            'page': page,
            'page_size': page_size,
            'exact_match': _format_keyword_results(keyword_results[start_idx:end_idx]),
            'code_structure': _format_semantic_results(
                unixcoder_results[start_idx:end_idx], 'UniXcoder'),
            'semantic': _format_semantic_results(
                sbert_results[start_idx:end_idx], 'SBERT'),
            'fused': _format_fused_results(fused[start_idx:end_idx]),
            'total_keyword': len(keyword_results),
            'total_unixcoder': len(unixcoder_results),
            'total_sbert': len(sbert_results),
            'total_fused': len(fused),
        }

        return jsonify(results)

    except Exception as e:
        app.logger.error(f"Search failed: {str(e)}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


if DEBUG_ROUTES_ENABLED:
    @app.route('/debug-search')
    def debug_search():
        """Debug endpoint (only available when dev.enable_debug_routes is true)"""
        try:
            query = request.args.get('q', 'test')
            keyword_results, unixcoder_results, sbert_results = compare_models(query, k=2)
            debug_info = {
                'query': query,
                'keyword_sample': keyword_results[:1] if keyword_results else [],
                'unixcoder_sample': unixcoder_results[:1] if unixcoder_results else [],
                'sbert_sample': sbert_results[:1] if sbert_results else []
            }
            return f"<pre>{str(html_escape(str(debug_info)))}</pre>"
        except Exception as e:
            return f"Error: {str(html_escape(str(e)))}"


# Helper functions

def _build_metadata_filter(data: Dict) -> Optional[Dict]:
    """Build a ChromaDB where clause from request filter parameters."""
    conditions = []

    file_filter = data.get('filter_file', '').strip()
    if file_filter:
        conditions.append({"file_path": {"$contains": file_filter}})

    func_filter = data.get('filter_function', '').strip()
    if func_filter:
        conditions.append({"function_name": func_filter})

    class_filter = data.get('filter_class', '').strip()
    if class_filter:
        conditions.append({"class_name": class_filter})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _resolve_file_path(file_path: str) -> str:
    """Resolve and sanitize file path using configuration"""
    if not file_path:
        raise ValueError("Empty file path")

    if os.path.isabs(file_path):
        return file_path

    project_root = os.path.dirname(os.path.dirname(__file__))

    try:
        from core.config import resolve_codebase_path
        codebase_path = resolve_codebase_path()
        codebase_relative = os.path.relpath(codebase_path, project_root)
    except Exception:
        codebase_relative = 'data/codebases/manuel_natcom/src'

    possible_paths = [
        os.path.join(project_root, file_path),
        os.path.join(project_root, codebase_relative, file_path),
        os.path.join(os.path.dirname(os.path.join(project_root, codebase_relative)), file_path),
        os.path.join(project_root, 'src', file_path),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)

    return os.path.abspath(os.path.join(project_root, codebase_relative, file_path))


def _parse_line_numbers(line: str, line_end: str) -> Tuple[int, int]:
    """Parse and validate line numbers"""
    try:
        start_line = int(line) if line else 1
        end_line = int(line_end) if line_end else start_line
    except (ValueError, TypeError):
        start_line = 1
        end_line = 1
    start_line = max(1, start_line)
    end_line = max(start_line, end_line)
    return start_line, end_line


def _generate_file_viewer_html(file_path: str, start_line: int, end_line: int) -> str:
    """Generate HTML for file viewer"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
    except Exception as e:
        raise Exception(f"Could not read file: {str(e)}")

    lines = file_content.split('\n')
    numbered_lines = []
    for i, line_content in enumerate(lines, 1):
        line_class = 'highlight-line' if start_line <= i <= end_line else ''
        numbered_lines.append({
            'number': i,
            'content': line_content,
            'class': line_class
        })

    if start_line == end_line:
        line_info = f"Highlighting line {start_line}"
    else:
        line_info = f"Highlighting lines {start_line}-{end_line}"

    escaped_basename = html_escape(os.path.basename(file_path))
    escaped_filepath = html_escape(file_path)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{escaped_basename} - Code Viewer</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                margin: 0; padding: 20px; background: #f8f9fa;
                font-size: 14px; line-height: 1.5;
            }}
            .container {{ max-width: 98%; margin: 0 auto; background: white;
                border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .header {{ background: #2c3e50; color: white; padding: 20px 25px;
                border-radius: 8px 8px 0 0; }}
            .file-name {{ font-size: 18px; font-weight: bold; margin-bottom: 8px; }}
            .file-path {{ font-size: 13px; color: #bdc3c7; word-break: break-all; }}
            .line-info {{ font-size: 13px; color: #f39c12; margin-top: 8px; }}
            .code-container {{ background: #ffffff; border-radius: 0 0 8px 8px;
                overflow: auto; max-height: 85vh; }}
            .code-line {{ display: flex; border-bottom: 1px solid #f1f2f6; min-height: 22px; }}
            .code-line:hover {{ background: #f8f9fa; }}
            .line-number {{ background: #f1f2f6; color: #666; padding: 4px 12px;
                text-align: right; min-width: 60px; border-right: 1px solid #e1e5e9;
                user-select: none; font-size: 12px; }}
            .line-content {{ padding: 4px 16px; white-space: pre; flex: 1; overflow-x: auto; }}
            .highlight-line {{ background: #fff3cd !important; border-left: 4px solid #f39c12; }}
            .highlight-line .line-number {{ background: #f39c12; color: white; font-weight: bold; }}
            .back-button {{ background: #3498db; color: white; padding: 10px 20px;
                border: none; border-radius: 6px; cursor: pointer; margin-top: 15px; font-size: 14px; }}
            .back-button:hover {{ background: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="file-name">{escaped_basename}</div>
                <div class="file-path">{escaped_filepath}</div>
                <div class="line-info">{line_info}</div>
                <button class="back-button" onclick="window.close()">Close</button>
            </div>
            <div class="code-container">
                {''.join(f'<div class="code-line {line_data["class"]}"><div class="line-number">{line_data["number"]}</div><div class="line-content">{html_escape(line_data["content"])}</div></div>' for line_data in numbered_lines)}
            </div>
        </div>
        <script>
            const firstHighlightedLine = document.querySelector('.highlight-line');
            if (firstHighlightedLine) {{
                firstHighlightedLine.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        </script>
    </body>
    </html>
    """


def _read_file_lines(file_path: str, start_line: int, end_line: int) -> str:
    """Read specific lines from a file"""
    try:
        if not os.path.isabs(file_path):
            file_path = _resolve_file_path(file_path)
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        start_line = max(1, int(start_line)) if start_line is not None else 1
        end_line = max(start_line, int(end_line)) if end_line is not None else start_line

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        if start_idx >= len(lines):
            return "Line number out of range"

        selected_lines = lines[start_idx:end_idx]
        formatted_lines = []
        for i, line in enumerate(selected_lines):
            line_num = start_line + i
            formatted_lines.append(f"{line_num:3}: {line.rstrip()}")
        return '\\n'.join(formatted_lines)

    except Exception as e:
        return f"Error reading file: {str(e)}"


def _format_result_common(result: Dict, i: int, score: float, score_type: str,
                          method_label: str) -> Optional[Dict]:
    """Shared formatting logic for all result types."""
    try:
        metadata = result.get('metadata', {})
        file_path = metadata.get('file_path') or result.get('file_path', 'Unknown')
        file_path = _convert_to_relative_path(file_path)

        content = result.get('content', '')
        function_name = (metadata.get('function_name') or result.get('function_name') or
                         _extract_function_name_from_content(content))

        if not function_name:
            class_name = metadata.get('class_name') or result.get('class_name')
            if class_name and class_name != 'Unknown':
                function_name = f"Class: {class_name}"
            else:
                function_name = f"Code Fragment {i + 1}"

        line_start = (metadata.get('start_line') or result.get('start_line') or
                      result.get('line_number', 1) or 1)
        line_end = metadata.get('end_line') or result.get('end_line', line_start) or line_start

        # For vector results, read from file; for keyword results, use stored content
        if score_type == 'cosine_similarity':
            code_content = _read_file_lines(file_path, line_start, line_end)
        else:
            code_content = content

        return {
            'function': function_name,
            'file': file_path,
            'code': code_content,
            'score': round(score, 3),
            'score_type': score_type,
            'has_score': True,
            'line_start': line_start,
            'line_end': line_end
        }
    except Exception as e:
        app.logger.warning(f"Error formatting {method_label} result: {e}")
        return None


def _format_keyword_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format keyword search results"""
    formatted = []
    for i, result in enumerate(results):
        score = result.get('score', 1.0)
        entry = _format_result_common(result, i, score, 'keyword_relevance', 'keyword')
        if entry:
            formatted.append(entry)
    return formatted


def _format_semantic_results(results: List[Dict[str, Any]], method_name: str) -> List[Dict[str, Any]]:
    """Format semantic search results (UniXcoder/SBERT)"""
    formatted = []
    for i, result in enumerate(results):
        distance = result.get('distance', 0)
        similarity = max(0, min(1, 1 - distance / 2))

        # Build function title
        function_name = result.get('function_name', '')
        class_name = result.get('class_name', '')
        if function_name and class_name:
            title = f"{class_name}::{function_name}"
        elif function_name:
            title = function_name
        elif class_name:
            title = f"class {class_name}"
        else:
            title = f"{method_name} Result {i + 1}"

        file_path = result.get('file_path', 'Unknown')
        if os.path.isabs(file_path):
            file_path = _convert_to_relative_path(file_path)

        line_start = result.get('start_line', 1)
        line_end = result.get('end_line', line_start)
        code_content = _read_file_lines(file_path, line_start, line_end)

        formatted.append({
            'function': title,
            'file': file_path,
            'code': code_content,
            'score': round(similarity, 3),
            'score_type': 'cosine_similarity',
            'has_score': True,
            'line_start': line_start,
            'line_end': line_end
        })
    return formatted


def _format_fused_results(results: List[Dict]) -> List[Dict[str, Any]]:
    """Format RRF-fused results."""
    formatted = []
    for i, result in enumerate(results):
        rrf_score = result.get('rrf_score', 0)
        entry = _format_result_common(result, i, rrf_score, 'rrf_fusion', 'fused')
        if entry:
            entry['source_methods'] = result.get('source_methods', [])
            formatted.append(entry)
    return formatted


def _convert_to_relative_path(file_path: str) -> str:
    """Convert absolute path to relative path from codebase root"""
    try:
        from core.config import resolve_codebase_path
        codebase_path = resolve_codebase_path()
        if file_path.startswith(str(codebase_path)):
            return os.path.relpath(file_path, codebase_path)
        parts = Path(file_path).parts
        if 'src' in parts:
            src_idx = parts.index('src')
            return '/'.join(parts[src_idx:])
        return file_path
    except Exception:
        if os.path.isabs(file_path):
            parts = Path(file_path).parts
            if 'src' in parts:
                src_idx = parts.index('src')
                return '/'.join(parts[src_idx:])
        return file_path


def _extract_function_name_from_content(content: str) -> Optional[str]:
    """Extract function name from code content"""
    import re
    func_patterns = [
        r'(?:void|int|double|float|char|bool|string|auto)\s+(\w+)\s*\(',
        r'(\w+)\s*\([^)]*\)\s*{',
        r'class\s+(\w+)',
        r'struct\s+(\w+)',
    ]
    for pattern in func_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting SearchYourCodes - Desktop Code Search Platform...")
    print("Access the interface at: http://localhost:8081")
    print("Press Ctrl+C to shutdown gracefully")

    try:
        debug_mode = app_config.get('debug', False) if 'app_config' in dir() else False
        app.run(debug=debug_mode, host='0.0.0.0', port=8081)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

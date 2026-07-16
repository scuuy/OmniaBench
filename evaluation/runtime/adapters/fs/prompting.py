"""
fs prompting：仅补充轻量提示，不强制复杂文件工作流。
"""

from __future__ import annotations


def _normalize_language_mode(language_mode: str | None) -> str:
    text = str(language_mode or "").strip().lower()
    if text.startswith("en"):
        return "en"
    return "zh"


def build_fs_prompt_addendum(language_mode: str = "zh") -> str:
    if _normalize_language_mode(language_mode) == "en":
        return """

Filesystem strategy addendum (use only when the task naturally needs files):
1. You may naturally use fs_list_dir / fs_read_file / fs_write_file / fs_move_path;
2. Do not force a complicated file workflow just to use file tools;
3. If the task involves file inputs, describe them as natural artifacts such as existing files, config files, spreadsheets, logs, or documents;
4. If multiple files are needed, organize them like a realistic scenario rather than stacking file operations mechanically.
"""
    return """

Filesystem 策略附加要求（仅在任务自然需要时使用）：
1. 你可以自然使用 fs_list_dir / fs_read_file / fs_write_file / fs_move_path 这些系统能力；
2. 不要为了使用文件工具而强行构造复杂流程；只读取一个文件也可能是自然任务；
3. 若任务涉及文件输入，可以把它表达成已有文件、配置文件、表格、日志、文档等自然场景；
4. 若需要多个文件，也应按真实场景组织，而不是机械堆叠文件操作。
"""


def build_fs_input_prompt_addendum(language_mode: str = "zh") -> str:
    if _normalize_language_mode(language_mode) == "en":
        return """
If the task naturally requires file inputs, you may declare files that need to be read.
Please provide:
1. file path;
2. file type;
3. the key information that must be obtained after reading (gold_read_result).
Do not generate the real file content here; the system will build the source files later in step6.5.
If the task does not require file inputs, do not output fs_inputs.

Additional hard constraints:
1. Output fs_inputs only when execution_chain truly uses fs_read_file, or uses fs_move_path that depends on existing files;
2. path_hint must not be a generic placeholder path like example.csv or test.json; it must be a realistic relative path;
3. Prefer shallow task-root relative paths such as `revision_note.json` or `q1_summary.md`; avoid deep directories unless semantically necessary;
4. path_hint must align with the file tool parameters in execution_chain, and at least one file tool step must directly reference it;
5. gold_read_result cannot be empty; it must contain the key values or excerpts the task truly depends on;
6. content_spec cannot be empty; for csv/tsv/xlsx include columns, for json/yaml/xml/toml/ini include key fields or structure hints;
7. For md/txt/html/docx/pdf/eml and similar text/document files, content_spec should include realistic clues such as `title`, `summary`, `sections`, `body_lines`, `style_hint`, `keywords`, or `excerpt`;
8. If a file is merely optional and not actually required by execution_chain, do not output fs_inputs.
"""
    return """
若任务自然涉及文件输入，可声明需要读取的文件。
请给出：
1. 文件路径；
2. 文件类型；
3. 读取后必须获得的关键信息（gold_read_result）。
不需要生成真实文件内容；系统会在后续 step6.5 统一构造原始文件。
若任务不需要文件输入，则不要输出 fs_inputs。

额外硬约束：
1. 只有当 execution_chain 中真实使用了 fs_read_file，或使用了依赖既有文件的 fs_move_path 时，才输出 fs_inputs；
2. path_hint 不能写成泛化占位路径（例如 example.csv、test.json）；必须是贴合场景的真实相对路径；
3. path_hint 默认优先使用 task 根目录下的浅层相对路径，例如 `revision_note.json`、`q1_summary.md`；除非语义确实要求，不要设计两层以上目录；
4. path_hint 必须与 execution_chain 里的文件工具参数保持一致，至少有一个文件工具步骤直接引用该路径；
5. gold_read_result 不能是空对象；至少要包含本任务真正依赖的 key/value 或可直接摘录的核心片段；
6. content_spec 不能留空；对于 csv/tsv/xlsx 至少给出 columns，对于 json/yaml/xml/toml/ini 至少给出关键字段或结构提示；
7. 对 md/txt/html/docx/pdf/eml 等文本或文档类文件，content_spec 应尽量提供真实内容线索，例如 `title`、`summary`、`sections`、`body_lines`、`style_hint`、`keywords`、`excerpt` 等，让后续文件生成更像真实资料；
8. 若只是“可能会用到文件”但执行链并未真正依赖它，则不要输出 fs_inputs。
"""

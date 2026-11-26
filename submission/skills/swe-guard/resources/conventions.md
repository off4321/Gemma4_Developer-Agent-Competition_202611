# SWE-Guard Repo Conventions

- **Sibling `ValueError` & Docstrings**: Match exact exception types, error strings, and docstrings of sibling helpers/validators (e.g., `SSE '<field>' must be a single line`).
- **Repo-Native Deprecations**: Use the repo's deprecation decorator (e.g., `@deprecated(..., category=FastAPIDeprecationWarning)`) instead of raw `warnings.warn(...)`, matching sibling signatures and `stacklevel`.
- **Pydantic v2 `AfterValidator(fn)`**: Define validator functions at **module level** taking 1 arg (`def _validate_x(v):`); never reference sibling class-body helpers inside `Annotated[..., AfterValidator(...)]` (causes class-body `NameError`).
- **Multi-File Parameter Propagation**: Trace parameters end-to-end before editing — e.g., `FastAPI.__init__` (`applications.py`) -> `APIRouter` (`routing.py`) -> `APIRoute` / `openapi/docs.py`. Use `ast_locate.py` to find all sites.
- **`src/`-Layout & `no_proxy`**: Run with `PYTHONPATH=src:.` so `src/`-layout repos (`requests`, etc.) import local edits instead of site-packages. For `no_proxy`, preserve `host_with_port` checks and `.domain` `lstrip(".")` suffix handling.
- **`rich` Module Map**: `ansi.py` (ANSI decoding), `markdown.py` (Markdown renderables), `file_proxy.py` (stdout/stderr proxy), `console.py` (`Console` capture/render), `cells.py` (CJK/character cell width).

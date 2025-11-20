"""
Code Validator - Validates Python code before execution

Checks for:
- Syntax errors
- Dangerous imports (os, subprocess, etc.)
- File system operations
- Network operations
- Execution safety
"""

import ast
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of code validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_code: Optional[str] = None


class CodeValidator:
    """
    Validates Python code for safe execution

    Blocks dangerous operations while allowing safe computation.
    """

    # Dangerous modules that are blocked
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'smtplib', 'telnetlib', 'shutil',
        'pathlib', 'tempfile', 'glob', 'pickle', 'marshal',
        'importlib', 'builtins', '__builtin__', 'ctypes',
        'multiprocessing', 'threading', 'asyncio'
    }

    # Dangerous built-in functions
    BLOCKED_BUILTINS = {
        'eval', 'exec', 'compile', '__import__', 'open',
        'input', 'raw_input', 'file', 'execfile'
    }

    def __init__(self, max_code_length: int = 10000):
        """
        Initialize validator

        Args:
            max_code_length: Maximum allowed code length in characters
        """
        self.max_code_length = max_code_length

    def validate(self, code: str) -> ValidationResult:
        """
        Validate Python code for safe execution

        Args:
            code: Python code to validate

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors = []
        warnings = []

        # Check code length
        if len(code) > self.max_code_length:
            errors.append(
                f"Code too long: {len(code)} chars (max: {self.max_code_length})"
            )
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

        # Check for empty code
        if not code.strip():
            errors.append("Code is empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

        # Parse and check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

        # Check for dangerous imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.BLOCKED_MODULES:
                        errors.append(
                            f"Blocked import: {alias.name} (security risk)"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.module in self.BLOCKED_MODULES:
                    errors.append(
                        f"Blocked import from: {node.module} (security risk)"
                    )

            # Check for dangerous built-in calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        errors.append(
                            f"Blocked function: {node.func.id}() (security risk)"
                        )

            # Check for file operations
            elif isinstance(node, ast.With):
                # Check if it's a file open
                if isinstance(node.items[0].context_expr, ast.Call):
                    if isinstance(node.items[0].context_expr.func, ast.Name):
                        if node.items[0].context_expr.func.id == 'open':
                            errors.append("File operations are not allowed")

        # Warnings for potentially slow operations
        code_lower = code.lower()
        if 'while true' in code_lower or 'while 1' in code_lower:
            warnings.append(
                "Infinite loop detected - execution will timeout after 30 seconds"
            )

        # If we have errors, code is invalid
        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

        # Code is valid
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            cleaned_code=code
        )

    def get_safe_imports(self) -> List[str]:
        """Get list of safe modules that are allowed"""
        return [
            'math', 'random', 'datetime', 'json', 'base64',
            'hashlib', 'hmac', 'statistics', 'collections',
            'itertools', 'functools', 'decimal', 'fractions',
            're', 'string', 'textwrap', 'unicodedata'
        ]

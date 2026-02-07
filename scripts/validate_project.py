#!/usr/bin/env python3
"""
Project Structure Validation Script

Validates that all required files are present and properly structured
for Project RLHFL.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


class ProjectValidator:
    """Validates project structure and files."""
    
    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.errors = []
        self.warnings = []
    
    def validate(self) -> bool:
        """Run all validation checks."""
        print("=" * 80)
        print("Project RLHFL - PROJECT VALIDATION")
        print("=" * 80)
        print()
        
        checks = [
            ("Required Files", self.check_required_files),
            ("Python Files", self.check_python_files),
            ("Docker Files", self.check_docker_files),
            ("Configuration", self.check_configuration),
            ("Documentation", self.check_documentation),
            ("Scripts", self.check_scripts),
        ]
        
        for name, check_func in checks:
            print(f"Checking {name}...")
            check_func()
            print()
        
        # Print summary
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        if self.errors:
            print(f"\n❌ Found {len(self.errors)} error(s):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validation checks passed!")
            return True
        elif not self.errors:
            print("\n✅ All critical checks passed (warnings only)")
            return True
        else:
            print("\n❌ Validation failed")
            return False
    
    def check_required_files(self):
        """Check that all required files exist."""
        required = [
            "docker-compose.yml",
            "Dockerfile.api",
            "Dockerfile.trainer",
            "README.md",
            "QUICKSTART.md",
            ".gitignore",
            ".dockerignore",
            "LICENSE",
            "Makefile",
        ]
        
        for file in required:
            if not (self.root / file).exists():
                self.errors.append(f"Missing required file: {file}")
            else:
                print(f"  ✓ {file}")
    
    def check_python_files(self):
        """Check Python module structure."""
        python_files = [
            "services/api/__init__.py",
            "services/api/main.py",
            "services/api/llm_engine.py",
            "services/api/memory_manager.py",
            "services/api/sentiment_analyzer.py",
            "services/trainer/__init__.py",
            "services/trainer/trainer_worker.py",
            "services/trainer/training_scheduler.py",
            "services/trainer/dataset_builder.py",
            "services/trainer/lora_trainer.py",
            "services/trainer/model_evaluator.py",
            "services/shared/__init__.py",
            "services/shared/config.py",
            "services/shared/models.py",
        ]
        
        for file in python_files:
            path = self.root / file
            if not path.exists():
                self.errors.append(f"Missing Python file: {file}")
            else:
                # Check for basic docstrings
                with open(path, 'r') as f:
                    content = f.read()
                    if '__init__.py' not in file and '"""' not in content[:500]:
                        self.warnings.append(f"No module docstring in {file}")
                    print(f"  ✓ {file}")
    
    def check_docker_files(self):
        """Check Docker configuration."""
        docker_files = [
            "Dockerfile.api",
            "Dockerfile.trainer",
            "docker-compose.yml",
            ".dockerignore",
        ]
        
        for file in docker_files:
            path = self.root / file
            if not path.exists():
                self.errors.append(f"Missing Docker file: {file}")
            else:
                print(f"  ✓ {file}")
    
    def check_configuration(self):
        """Check configuration files."""
        config_files = [
            "volumes/config/system_config.yaml",
            ".env.example",
        ]
        
        for file in config_files:
            path = self.root / file
            if not path.exists():
                self.errors.append(f"Missing config file: {file}")
            else:
                print(f"  ✓ {file}")
        
        # Check for placeholder directories
        placeholder_dirs = [
            "volumes/models",
            "volumes/data",
            "volumes/checkpoints",
        ]
        
        for dir_path in placeholder_dirs:
            path = self.root / dir_path
            if not path.exists():
                self.warnings.append(f"Missing directory: {dir_path}")
            else:
                print(f"  ✓ {dir_path}/")
    
    def check_documentation(self):
        """Check documentation files."""
        docs = [
            ("README.md", 5000),           # Should be substantial
            ("QUICKSTART.md", 2000),
            ("TECHNICAL.md", 5000),
            ("CHANGELOG.md", 500),
            ("CONTRIBUTING.md", 2000),
            ("LICENSE", 500),
        ]
        
        for file, min_size in docs:
            path = self.root / file
            if not path.exists():
                self.errors.append(f"Missing documentation: {file}")
            else:
                size = path.stat().st_size
                if size < min_size:
                    self.warnings.append(
                        f"{file} seems short ({size} bytes, expected >{min_size})"
                    )
                print(f"  ✓ {file} ({size} bytes)")
    
    def check_scripts(self):
        """Check utility scripts."""
        scripts = [
            ("scripts/download_models.sh", True),   # Should be executable
            ("scripts/health_check.py", True),
            ("scripts/example_client.py", True),
        ]
        
        for file, should_be_executable in scripts:
            path = self.root / file
            if not path.exists():
                self.errors.append(f"Missing script: {file}")
            else:
                if should_be_executable and not os.access(path, os.X_OK):
                    self.warnings.append(f"Script not executable: {file}")
                print(f"  ✓ {file}")


def main():
    """Run validation."""
    validator = ProjectValidator()
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

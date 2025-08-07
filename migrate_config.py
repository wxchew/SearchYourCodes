#!/usr/bin/env python3
"""
Configuration Migration Script for CodeFinder

This script helps migrate from the old JSON-based configuration system 
to the new unified YAML configuration system.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any

def migrate_json_configs():
    """Migrate old JSON configuration files to the new system."""
    print("🔄 Starting configuration migration...")
    
    config_dir = Path("config")
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    # Create backup directory
    backup_dir = config_dir / "backup_old_configs"
    backup_dir.mkdir(exist_ok=True)
    
    # Files to migrate (move to data directory as state files)
    json_files = [
        "embedding_model.json",
        "embedding_models.json", 
        "unixcoder_models.json",
        "test_embedding_models.json",
        "test_raw_embedding_models.json"
    ]
    
    migrated_files = []
    
    for json_file in json_files:
        json_path = config_dir / json_file
        
        if json_path.exists():
            print(f"  📄 Found {json_file}")
            
            # Backup the original file
            backup_path = backup_dir / json_file
            shutil.copy2(json_path, backup_path)
            print(f"    ✅ Backed up to {backup_path}")
            
            # For the main model config files, convert to new format
            if json_file == "unixcoder_models.json":
                try:
                    with open(json_path, 'r') as f:
                        old_config = json.load(f)
                    
                    # Convert to new model state format
                    new_state = convert_to_model_state(old_config)
                    
                    # Save as model_state.json in processed directory
                    state_file = processed_dir / "model_state.json"
                    processed_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(state_file, 'w') as f:
                        json.dump(new_state, f, indent=2)
                    
                    print(f"    🔄 Converted to {state_file}")
                    migrated_files.append((json_path, state_file))
                    
                except Exception as e:
                    print(f"    ⚠️  Error converting {json_file}: {e}")
            
            # Move original to backup (don't delete yet, just rename)
            old_path = json_path.with_suffix('.json.old')
            json_path.rename(old_path)
            print(f"    📦 Moved original to {old_path}")
    
    if migrated_files:
        print(f"\n✅ Migration completed!")
        print(f"  📊 Migrated {len(migrated_files)} configuration files")
        print(f"  💾 Backups saved in: {backup_dir}")
        print(f"\n📝 Next steps:")
        print(f"  1. Verify your config.yaml has the correct model configurations")
        print(f"  2. Test the system with: python main.py --config")
        print(f"  3. If everything works, you can safely delete the .old files")
    else:
        print("  ℹ️  No JSON configuration files found to migrate")
    
    return migrated_files

def convert_to_model_state(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old unixcoder_models.json format to new model_state.json format."""
    
    new_state = {
        "models": {},
        "sample_info": old_config.get("sample_info", {})
    }
    
    # Convert model configurations
    if "models" in old_config:
        for old_name, old_model in old_config["models"].items():
            # Map old model names to new names
            if "Sentence-BERT_MiniLM" in old_name or "MiniLM" in old_name:
                new_name = "sbert"
            elif "UniXcoder" in old_name or "unixcoder" in old_name:
                new_name = "unixcoder"
            else:
                new_name = old_name.lower().replace("-", "_")
            
            # Convert model config format
            new_state["models"][new_name] = {
                "model_name": old_model.get("model_name", ""),
                "model_type": old_model.get("model_type", ""),
                "embedding_file": old_model.get("embedding_file", ""),
                "embedding_shape": old_model.get("embedding_shape", []),
                "device": old_model.get("device", "auto"),
                "pooling_method": old_model.get("pooling_method", "built-in")
            }
    
    return new_state

def check_migration_status():
    """Check if migration is needed and show current status."""
    config_dir = Path("config")
    processed_dir = Path("data/processed")
    
    print("🔍 Checking configuration migration status...")
    
    # Check for old JSON files
    old_files = []
    for json_file in ["unixcoder_models.json", "embedding_models.json"]:
        json_path = config_dir / json_file
        if json_path.exists():
            old_files.append(json_file)
    
    # Check for new state file
    state_file = processed_dir / "model_state.json"
    has_new_state = state_file.exists()
    
    # Check main config
    config_file = Path("config.yaml")
    has_main_config = config_file.exists()
    
    print(f"  📋 Main config.yaml: {'✅ Found' if has_main_config else '❌ Missing'}")
    print(f"  📊 Model state file: {'✅ Found' if has_new_state else '❌ Missing'}")
    print(f"  📄 Old JSON configs: {len(old_files)} found {old_files if old_files else ''}")
    
    if old_files and not has_new_state:
        print(f"\n💡 Migration recommended! Run with --migrate to convert old configuration files.")
        return False
    elif has_new_state and has_main_config:
        print(f"\n✅ Configuration system is up to date!")
        return True
    else:
        print(f"\n⚠️  Configuration system needs attention.")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate CodeFinder configuration files")
    parser.add_argument("--migrate", action="store_true", help="Perform the migration")
    parser.add_argument("--check", action="store_true", help="Check migration status only")
    
    args = parser.parse_args()
    
    if args.migrate:
        migrate_json_configs()
    elif args.check:
        check_migration_status()
    else:
        # Default: check status and offer to migrate
        needs_migration = not check_migration_status()
        if needs_migration:
            response = input("\n❓ Would you like to migrate now? (y/N): ")
            if response.lower().startswith('y'):
                migrate_json_configs()

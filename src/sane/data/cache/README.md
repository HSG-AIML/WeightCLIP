# Cache Utilities

Tools for inspecting and managing cache directories created by `CachedWindowedDataset`.

## Directory Structure

Cache files are stored in a nested layout under the base cache directory:

```
base_cache_dir/
  {dataset}_{config_hash}/
    checkpoint/
      {split_name}/
        cache_*.pt
    window/
      {split_name}/
        cache_*.pt
```

Each top-level subdirectory corresponds to one cache configuration (a unique combination of dataset and tokenizer/windowing settings), identified by a hash.

## Functions

### `list_all_cache_configs(base_cache_dir)`

Returns a list of all cache configurations found under `base_cache_dir`, sorted by size (largest first). Each entry is a dict with:

- `config_hash` — directory name (dataset + hash)
- `cache_dir` — full path
- `num_files` — number of `.pt` files
- `total_size_mb` / `total_size_gb` — disk usage

```python
from sane.data.cache.cache_utils import list_all_cache_configs

configs = list_all_cache_configs("/local/cache/tokens")
for c in configs:
    print(c["config_hash"], f"{c['total_size_gb']:.2f} GB", f"({c['num_files']} files)")
```

### `print_cache_summary(base_cache_dir)`

Prints a human-readable summary of all configs and total disk usage.

```python
from sane.data.cache.cache_utils import print_cache_summary

print_cache_summary("/local/cache/tokens")
```

### `get_cache_size(cache_dir)`

Returns the disk usage of a single config directory as a dict with keys `bytes`, `mb`, `gb`.

```python
from sane.data.cache.cache_utils import get_cache_size

size = get_cache_size("/local/cache/tokens/default_abc123")
print(f"{size['gb']:.3f} GB")
```

### `clear_cache_config(cache_dir, dry_run=False)`

Deletes a single config directory. Includes a safety check — refuses to delete a directory that contains no `cache_*.pt` files. Pass `dry_run=True` to preview without deleting.

```python
from sane.data.cache.cache_utils import clear_cache_config

# Preview
clear_cache_config("/local/cache/tokens/default_abc123", dry_run=True)

# Delete
clear_cache_config("/local/cache/tokens/default_abc123")
```

### `clear_old_caches(base_cache_dir, keep_latest_n=1, dry_run=False)`

Removes all but the `keep_latest_n` largest config directories. Useful for freeing disk space after switching to a new dataset or tokenizer config.

```python
from sane.data.cache.cache_utils import clear_old_caches

# Keep the 2 largest configs, remove the rest
clear_old_caches("/local/cache/tokens", keep_latest_n=2, dry_run=True)
clear_old_caches("/local/cache/tokens", keep_latest_n=2)
```

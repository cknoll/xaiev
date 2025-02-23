This file serves as easy to use internal issue and idea tracker.

# Issues (and required features)

- [ ] #i9: resolve conflict between model_name and model_full_name (eg. simple_cnn vs simple_cnn_1_1) in README and code
- [ ] #i8: implement unittests for steps (check existence of files)
- [ ] #i7: add `xaiev eval ...` commands
- [x] #i6: simplify CI debugging
- [x] #i5: rename atsds-directories in README and scripts
- [x] #i4: create small publishable test-dataset (atsds_demo)
- [x] #i3: add --limit flag to limit the processing of images (for faster testing)
- [x] #i2: make inference work from `xaiev --inference simple_cnn_1_1`
- [x] #i1: add `xaiev --bootstrap` (to create .env file in current working directory)
    - motivation: this simplifies installing from pypi

# Ideas

- `xaiev --bootstrap /path/to/data/dir` would be nice (no need to manually edit the .env file after creation)
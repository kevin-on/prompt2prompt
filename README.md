# P2P (Prompt-to-Prompt) Implementation

This repository contains a refactored Python implementation of the [Prompt-to-Prompt Image Editing](https://github.com/google/prompt-to-prompt) project, originally developed by Google Research.

## Motivation

The original Prompt-to-Prompt repository provides an innovative approach to image editing using diffusion models, allowing users to make precise modifications to generated images by editing input prompts. However, there were two main challenges with the original implementation:

1. **Code Structure**: The original code was implemented in Jupyter notebooks, which while great for demonstrations, isn't ideal for production use or integration into larger projects. This repository refactors the code into a proper Python project structure for better maintainability and usability.

2. **Compatibility Issues**: Recent versions of the `diffusers` library introduced breaking changes that affected the original implementation. This repository includes fixes for these compatibility issues, particularly addressing problems with newer diffusers versions as discussed in [this issue](https://github.com/google/prompt-to-prompt/issues/90).

### Running the Project

The main script for running prompt-to-prompt image editing is located in the `scripts` directory. To run the project, execute the following command from the root directory of the project:

```bash
python scripts/run.py
```

This will run the default image editing pipeline with pre-configured examples.

## Acknowledgments

- Original implementation by [Google Research](https://github.com/google/prompt-to-prompt)
- Community contributions for diffusers compatibility fixes

## License

This project follows the same licensing as the original repository (Apache 2.0).

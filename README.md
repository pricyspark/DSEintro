# ImageMorph

ImageMorph reconstructs a target image by rearranging the pixels of a source
image using Hungarian matching.

## Installation

Create and activate the Conda environment:

```shell
conda env create -f environment.yml
conda activate imagemorph
```

## Usage

Note: Please don't try and break or edge test the CLI. Adding QOL
features ended up being way more work than I estimated and took much
longer than the actual backend.

Pass the source image followed by the target image:

```shell
python cli.py images/source.jpg images/target.jpg
```

The result is written to `reconstructed.jpg` by default. Use `--output` to
choose another path:

```shell
python cli.py images/source.jpg images/target.jpg --output result.png
```

Use a downsample factor, desired longest edge, or target runtime to choose the
problem size:

```shell
python cli.py images/source.jpg images/target.jpg --downsample 20
python cli.py images/source.jpg images/target.jpg --edge-length 100
python cli.py images/source.jpg images/target.jpg --target-runtime 5
```

Target runtime is specified in seconds. ImageMorph runs a short benchmark and
selects an edge length whose estimated matching time fits that runtime. Use
`--benchmark-seconds` to change the default 10-second benchmark budget:

```shell
python cli.py images/source.jpg images/target.jpg --target-runtime 5 --benchmark-seconds 3
```

To estimate the matching time before reconstruction, optionally setting the
benchmark duration in seconds:

```shell
python cli.py images/source.jpg images/target.jpg --edge-length 100 --estimate-runtime
python cli.py images/source.jpg images/target.jpg --edge-length 100 --estimate-runtime --benchmark-seconds 5
```

Run `python cli.py --help` to see all available options.

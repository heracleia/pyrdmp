# pyrdmp

Python Library for Dynamic Movement Primitives with Reinforcement Learning

## Dependencies

Python package requirements can be installed via pip with: `pip install -r requirements.txt`

Additionally, you will need the python3-tk package which can be installed on Ubuntu with:
`sudo apt install python3-tk`

## Running

Install the `pyrdmp` package locally with: `pip install -e .` if developing the library.
Otherwise, releases can be install with `pip install pyrdmp` or by cloning the repository and: `pip install .`

## Examples

You can run examples by navigating to `examples/` 
and running `./ex_dmp_adaptation.py` 

`ex_dmp_adaptation.py` includes an argparser. 
A description of the available commands can be found with: `python ex_dmp_adaptation.py --help`

## Notes

Tested on Ubuntu 16.04, 14.04, and 12.04

## Citing

```
@misc{Theofanidis2019,
  author = {Theofanidis, Michail and Cloud, Joe and Brady, James},
  title = {pyrdmp: python library for dynamic movement primitives with reinforcement learning},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/heracleia/pyrdmp}},
  commit = {cfaeae5d0634e98433c1121ed2f69ca69483e0b1}
}


```


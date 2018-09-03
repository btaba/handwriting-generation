# handwriting-generation

Train a network to generate handwriting both unconditionally and conditionally based on text input. This code is based on [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf).


## How do I run it?

Run the jupyter notebook in Docker:

* `source run_docker.sh`
* Follow the link in the terminal to run `notebooks/results.ipynb` to generate some samples!


Or run it locally:

* `source run_local.sh`


## Results

The results can be run in a jupyter notebook using `./run_docker.sh` or `./run_local.sh`


#### Unconditionally Generated Samples

|   |
|---|
| ![unconditional 1](./assets/unconditional_stroke1.png)  |
| ![unconditional 2](./assets/unconditional_stroke2.png)  |
| ![unconditional 3](./assets/unconditional_stroke3.png)  |



#### Conditionally Generated Samples

`Baruch Tabanpour was here `


|   |
|---|
| ![baruch tabanpour](./assets/conditional_stroke2v1.png)  |
| ![baruch tabanpour](./assets/conditional_stroke2v2.png)  |



`france will win the world cup`


|   |
|---|
| ![the world cup](./assets/conditional-stroke4v1.png)  |
| ![the world cup](./assets/conditional-stroke4v2.png)  |



`welcome to my house`


|   |
|---|
| ![welcome to my house](./assets/conditional_stroke1v1.png)  |
| ![welcome to my house](./assets/conditional_stroke1v2.png)  |



`welcome to canada`


|   |
|---|
| ![welcome to canada](./assets/conditional-stroke3v1.png)  |
| ![welcome to canada](./assets/conditional-stroke3v2.png)  |



## Training

Set these env vars to point to your data and model directories:

* `HANDWRITING_GENERATION_DATA_DIR`
* `HANDWRITING_GENERATION_MODEL_DIR`

Then you can train one of the models:

* Unconditional stroke generation:
    - `python unconditional_stroke_model.py ../models/unconditional-stroke-model-1/`
* Conditional stroke generation:
    - `python conditional_stroke_model.py ../models/conditional-stroke-model-1/`

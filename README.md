# Rust AI

**Rust AI** is a deep learning framework made in rust. the library let's you define the structure of a neural neural network and then train it using backpropagation.

## Try example in the stable release

donwload Rust_AI.exe from the stable release

run it

## Install

clone the repository or download the V1 release

```console
git clone https://github.com/TecnoSasso/RustAI.git
```

**IF NOT ALREADY DONE** install rust and cargo

```console
curl https://sh.rustup.rs -sSf | sh
```

if everything goes well you should see

```console
Rust is installed now. Great!
```

and that's it, you have an example file in `V1/src/main.rs` and in `V2/src/main.rs`. If you wish to run V2 you might have to install the CUDA toolkit (only works with Nvidia GPUs for now)

## Usage

**THIS IS ONLY VALID FOR V1**, for V2 see the V2/src/main.rs example

declare the model:

```rs
let mut model = NN {
    layers: vec![
        Layer::linear(2, Sigmoid),
        Layer::linear(2, Tanh),
        Layer::linear(2, SoftMax)
        ],
        loss: CrossEntropy
    };
```

compile it

```rs
model.compile();
```

prepare the training data

```rs
let train_data: Vec<(DVector<f64>, DVector<f64>)> = [
    (dvector![0.0, 0.0], dvector![0.0, 1.0]),
    (dvector![0.0, 1.0], dvector![1.0, 0.0]),
    (dvector![1.0, 0.0], dvector![1.0, 0.0]),
    (dvector![1.0, 1.0], dvector![0.0, 1.0])
].to_vec();
```

then train it!

```rs
for _ in 0..1000{
    model.train(&train_data, 4, BackProp, 1.0);
}
```

result:

```console
...
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00173
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00173
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00172
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00172
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00172
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00172
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00172
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00171
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00171
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00171
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00171
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00171
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00171
Batch 1 of 1 completed  Accuracy: 100.00,  Cost: 0.00170
```

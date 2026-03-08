# jmicrograd

A Java port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a tiny scalar-valued autograd engine with a small neural network library on top.

## Overview

jmicrograd implements reverse-mode automatic differentiation (backpropagation) over a dynamically built computation graph. It supports basic operations (add, multiply, power) and activation functions (ReLU, tanh, sigmoid).

## Building

```bash
mvn package
```

## Usage

```java
import static com.github.kwart.jmicrograd.Value.value;

Value a = value(-4.0);
Value b = value(2.0);
Value c = a.add(b).mul(b.pow(3));
c.backward();

System.out.println(a.getGrad()); // dc/da
System.out.println(b.getGrad()); // dc/db
```

## Acknowledgements

Based on [micrograd](https://github.com/karpathy/micrograd) by [Andrej Karpathy](https://github.com/karpathy). See also his excellent [video explanation](https://www.youtube.com/watch?v=VMj-3S1tku0).

package com.github.kwart.jmicrograd;

import java.util.function.Function;

public class Layer implements Function<Value[], Value[]> {

    private final Neuron[] neurons;

    public Layer(int inputs, int outputs) {
        neurons = new Neuron[outputs];
        for (int i = 0; i < outputs; i++) {
            neurons[i] = new Neuron(inputs);
        }
    }

    @Override
    public Value[] apply(Value[] x) {
        if (x == null || x.length != neurons.length)
            throw new IllegalArgumentException(String.format("Lengths have to be equal (%d != %d)", x.length, neurons.length));

        Value[] out = new Value[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            out[i] = neurons[i].apply(x);
        }
        return out;
    }

}

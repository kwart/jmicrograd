package com.github.kwart.jmicrograd;

import java.util.function.Function;

public class Layer implements Function<Value[], Value[]> {

    private final Neuron[] neurons;

    public Layer(int inputs, int outputs) {
        if (inputs <= 0 || outputs <= 0)
            throw new IllegalArgumentException(
                    String.format("Inputs (%d) and outputs (%d) have to be positive numbers", inputs, outputs));
        neurons = new Neuron[outputs];
        for (int i = 0; i < outputs; i++) {
            neurons[i] = new Neuron(inputs);
        }
    }

    @Override
    public Value[] apply(Value[] x) {
        if (x == null || x.length == 0)
            throw new IllegalArgumentException("Non-empty array expected");

        Value[] out = new Value[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            out[i] = neurons[i].apply(x);
        }
        return out;
    }

    public Value[] parameters() {
        Value[] out = new Value[parametersLength()];
        int npar = neurons[0].parametersLength();
        for (int i = 0; i < neurons.length; i++) {
            Neuron n = neurons[i];
            System.arraycopy(n.parameters(), 0, out, i * npar, npar);
        }
        return out;
    }

    public int parametersLength() {
        return neurons.length * neurons[0].parametersLength();
    }
}

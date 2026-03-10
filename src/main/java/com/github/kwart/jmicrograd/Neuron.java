package com.github.kwart.jmicrograd;

import static com.github.kwart.jmicrograd.Value.value;

import java.util.function.Function;
import java.util.random.RandomGenerator;

public class Neuron implements Function<Value[], Value> {

    // TODO as the constructor parameter
    private static final RandomGenerator rnd = RandomGenerator.getDefault();

    private final Value[] weights;
    private final Value bias;

    public Neuron(int inputs) {
        // TODO as the parameter
        WeightInit init = WeightInit.UNIFORM;
        if (inputs <= 0)
            throw new IllegalArgumentException("inputs must be positive");

        weights = new Value[inputs];
        for (int i = 0; i < inputs; i++) {
            weights[i] = value(init.nextWeight(rnd, inputs));
        }

        bias = value(init.nextWeight(rnd, inputs));
    }

    @Override
    public Value apply(Value[] x) {
        if (x == null || x.length != weights.length)
            throw new IllegalArgumentException(String.format("Lengths have to be equal (%d != %d)", x.length, weights.length));
        Value act = bias;
        for (int i = 0; i < x.length; i++) {
            act = act.add(x[i].mul(weights[i]));
        }
        Value out = act.tanh();
        return out;
    }

    public Value[] parameters() {
        Value[] out = new Value[weights.length + 1];
        System.arraycopy(weights, 0, out, 0, weights.length);
        out[weights.length] = bias;
        return out;
    }

    public int parametersLength() {
        return weights.length + 1;
    }
    
}

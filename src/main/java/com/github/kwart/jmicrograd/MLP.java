package com.github.kwart.jmicrograd;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Multilayer Perceptron (MLP)
 */
public class MLP implements Function<Value[], Value[]> {

    private final Layer[] layers;

    public MLP(int inputs, int[] outputs) {
        layers = new Layer[outputs.length];
        for (int i = 0; i < layers.length; i++) {
            inputs = i == 0 ? inputs : outputs[i - 1];
            layers[i] = new Layer(inputs, outputs[i]);
        }
    }

    @Override
    public Value[] apply(Value[] x) {
        Value[] out = x;
        for (Layer layer : layers) {
            out = layer.apply(out);
        }
        return out;
    }

    public Value[] parameters() {
        Value[] out = new Value[parametersLength()];
        int idx = 0;
        for (int i = 0; i < layers.length; i++) {
            Layer l = layers[i];
            int lpar = l.parametersLength();
            System.arraycopy(l.parameters(), 0, out, idx, lpar);
            idx += lpar;
        }
        return out;
    }

    public int parametersLength() {
        return Arrays.stream(layers).mapToInt(Layer::parametersLength).sum();
    }

    public void zeroGrad() {
        for (Value parameter : parameters()) {
            parameter.setGrad(0.);
        }
    }

}

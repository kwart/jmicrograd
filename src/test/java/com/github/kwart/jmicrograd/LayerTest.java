package com.github.kwart.jmicrograd;

import static com.github.kwart.jmicrograd.Value.value;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class LayerTest {

    @Test
    void constructorRejectsZeroInputs() {
        assertThatThrownBy(() -> new Layer(0, 3))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void constructorRejectsZeroOutputs() {
        assertThatThrownBy(() -> new Layer(3, 0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void constructorRejectsNegativeInputs() {
        assertThatThrownBy(() -> new Layer(-1, 3))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void constructorRejectsNegativeOutputs() {
        assertThatThrownBy(() -> new Layer(3, -2))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void parametersLengthIsCorrect() {
        // 3 inputs, 4 outputs -> 4 neurons, each with 3 weights + 1 bias = 16
        Layer layer = new Layer(3, 4);
        assertThat(layer.parametersLength()).isEqualTo(16);
        assertThat(layer.parameters()).hasSize(16);
    }

    @Test
    void parametersLengthSingleNeuron() {
        Layer layer = new Layer(2, 1);
        assertThat(layer.parametersLength()).isEqualTo(3);
    }

    @Test
    void applyReturnsCorrectOutputSize() {
        Layer layer = new Layer(3, 4);
        Value[] x = { value(1.0), value(2.0), value(3.0) };
        Value[] out = layer.apply(x);
        assertThat(out).hasSize(4);
    }

    @Test
    void applyOutputsAreInTanhRange() {
        Layer layer = new Layer(3, 2);
        Value[] x = { value(0.5), value(-0.5), value(0.1) };
        Value[] out = layer.apply(x);
        for (Value v : out) {
            assertThat(v.getData()).isBetween(-1.0, 1.0);
        }
    }

    @Test
    void applyRejectsNullInput() {
        Layer layer = new Layer(2, 3);
        assertThatThrownBy(() -> layer.apply(null))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void applyRejectsEmptyInput() {
        Layer layer = new Layer(2, 3);
        assertThatThrownBy(() -> layer.apply(new Value[0]))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void applyRejectsWrongInputLength() {
        Layer layer = new Layer(3, 2);
        // Input length 2 doesn't match the 3 inputs expected by each neuron
        Value[] x = { value(1.0), value(2.0) };
        assertThatThrownBy(() -> layer.apply(x))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void applySupportsBackpropagation() {
        Layer layer = new Layer(2, 3);
        Value[] x = { value(1.0), value(2.0) };
        Value[] out = layer.apply(x);

        // Sum outputs and backpropagate
        Value loss = out[0].add(out[1]).add(out[2]);
        loss.backward();

        for (Value p : layer.parameters()) {
            assertThat(p.getGrad()).isNotNaN();
        }
        for (Value xi : x) {
            assertThat(xi.getGrad()).isNotNaN();
        }
    }

    @Test
    void parametersContainAllNeuronParameters() {
        Layer layer = new Layer(2, 3);
        Value[] params = layer.parameters();
        // 3 neurons * (2 weights + 1 bias) = 9
        assertThat(params).hasSize(9);
        for (Value p : params) {
            assertThat(p).isNotNull();
            assertThat(p.getData()).isNotNaN();
        }
    }
}

package com.github.kwart.jmicrograd;

import static com.github.kwart.jmicrograd.Value.value;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class MLPTest {

    private static final double TOL = 1e-6;

    @Test
    void parametersLengthIsCorrect() {
        // MLP(3, [4, 4, 1])
        // Layer 0: 3 inputs, 4 outputs -> 4 * (3+1) = 16
        // Layer 1: 4 inputs, 4 outputs -> 4 * (4+1) = 20
        // Layer 2: 4 inputs, 1 output  -> 1 * (4+1) = 5
        // Total = 41
        MLP mlp = new MLP(3, new int[]{4, 4, 1});
        assertThat(mlp.parametersLength()).isEqualTo(41);
        assertThat(mlp.parameters()).hasSize(41);
    }

    @Test
    void singleLayerMLP() {
        // MLP(2, [1]) -> single layer with 2 inputs, 1 output
        MLP mlp = new MLP(2, new int[]{1});
        assertThat(mlp.parametersLength()).isEqualTo(3);
    }

    @Test
    void applyProducesCorrectOutputSize() {
        MLP mlp = new MLP(3, new int[]{4, 4, 1});
        Value[] x = { value(1.0), value(2.0), value(3.0) };
        Value[] out = mlp.apply(x);
        assertThat(out).hasSize(1);
    }

    @Test
    void applyMultipleOutputs() {
        MLP mlp = new MLP(2, new int[]{3, 2});
        Value[] x = { value(0.5), value(-0.5) };
        Value[] out = mlp.apply(x);
        assertThat(out).hasSize(2);
        for (Value v : out) {
            assertThat(v.getData()).isBetween(-1.0, 1.0);
        }
    }

    @Test
    void forwardAndBackward() {
        MLP mlp = new MLP(2, new int[]{3, 1});
        Value[] x = { value(1.0), value(-1.0) };
        Value[] out = mlp.apply(x);

        out[0].backward();

        // All parameters should have gradients set
        for (Value p : mlp.parameters()) {
            assertThat(p.getGrad()).isNotNaN();
        }
        // Input gradients should be set
        for (Value xi : x) {
            assertThat(xi.getGrad()).isNotNaN();
        }
    }

    @Test
    void zeroGradResetsAllGradients() {
        MLP mlp = new MLP(2, new int[]{3, 1});
        Value[] x = { value(1.0), value(-1.0) };
        Value[] out = mlp.apply(x);
        out[0].backward();

        // Verify some gradients are non-zero before zeroGrad
        boolean hasNonZero = false;
        for (Value p : mlp.parameters()) {
            if (p.getGrad() != 0.0) {
                hasNonZero = true;
                break;
            }
        }
        assertThat(hasNonZero).isTrue();

        mlp.zeroGrad();

        for (Value p : mlp.parameters()) {
            assertThat(p.getGrad()).isEqualTo(0.0);
        }
    }

    @Test
    void trainingStepReducesLoss() {
        MLP mlp = new MLP(1, new int[]{2, 1});

        double target = 0.8;

        // Compute initial loss
        Value[] out1 = mlp.apply(new Value[]{ value(0.5) });
        double initialLoss = out1[0].add(value(-target)).pow(2).getData();

        // Do a few training steps
        double lr = 0.1;
        for (int step = 0; step < 10; step++) {
            mlp.zeroGrad();
            Value[] outStep = mlp.apply(new Value[]{ value(0.5) });
            Value loss = outStep[0].add(value(-target)).pow(2);
            loss.backward();

            for (Value p : mlp.parameters()) {
                p.setData(p.getData() - lr * p.getGrad());
            }
        }

        // Compute final loss
        Value[] outFinal = mlp.apply(new Value[]{ value(0.5) });
        double finalLoss = outFinal[0].add(value(-target)).pow(2).getData();

        assertThat(finalLoss).isLessThan(initialLoss);
    }

    @Test
    void parametersAreNotNull() {
        MLP mlp = new MLP(3, new int[]{4, 4, 1});
        for (Value p : mlp.parameters()) {
            assertThat(p).isNotNull();
            assertThat(p.getData()).isNotNaN();
        }
    }

    @Test
    void multipleForwardPassesProduceSameResult() {
        MLP mlp = new MLP(2, new int[]{3, 1});
        Value[] out1 = mlp.apply(new Value[]{ value(1.0), value(2.0) });
        Value[] out2 = mlp.apply(new Value[]{ value(1.0), value(2.0) });
        assertThat(out1[0].getData()).isCloseTo(out2[0].getData(), within(TOL));
    }

    @Test
    void singleLayerForwardPass() {
        MLP mlp = new MLP(3, new int[]{2});
        Value[] x = { value(0.3), value(0.7), value(-0.2) };
        Value[] out = mlp.apply(x);
        assertThat(out).hasSize(2);
        for (Value v : out) {
            assertThat(v.getData()).isBetween(-1.0, 1.0);
        }
    }
}

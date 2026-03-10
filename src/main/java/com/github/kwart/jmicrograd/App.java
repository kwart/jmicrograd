package com.github.kwart.jmicrograd;

import static com.github.kwart.jmicrograd.Value.value;

import java.util.Arrays;
import java.util.Random;

/**
 * The App!
 */
public class App {

    public static void main(String[] args) {
        valueDemo();
        System.out.println();

        mlpDemo();
        System.out.println();

        demo();
    }

    /**
     * MLP demo ported from micrograd - trains a small network on 4 examples using MSE loss.
     */
    public static void mlpDemo() {
        System.out.println("=== MLP Demo: Learning 4 Target Values ===");
        System.out.println("Network architecture: MLP(3, [4, 4, 1])");
        System.out.println("Training a 3-input, 1-output network with two hidden layers of 4 neurons each.");
        System.out.println();

        Value[] x = { value(2.0), value(3.0), value(-1.0) };
        MLP n = new MLP(3, new int[] { 4, 4, 1 });
        System.out.println("Number of parameters: " + n.parametersLength());
        System.out.println("Single forward pass for x=[2.0, 3.0, -1.0]: " + Arrays.toString(n.apply(x)));
        System.out.println();

        Value[][] xs = { { value(2.0), value(3.0), value(-1.0) }, { value(3.0), value(-1.0), value(0.5) },
                { value(0.5), value(1.0), value(1.0) }, { value(1.0), value(1.0), value(-1.0) }, };
        double[] ys = { 1.0, -1.0, -1.0, 1.0 }; // desired targets

        System.out.println("Training data (4 samples), desired targets: [1.0, -1.0, -1.0, 1.0]");
        System.out.println("Loss function: MSE (mean squared error)");
        System.out.println("Optimizer: SGD with learning rate 0.1");
        System.out.println();
        System.out.println("Training for 20 steps...");

        for (int k = 0; k < 20; k++) {
            // forward pass
            Value[] ypred = new Value[xs.length];
            for (int i = 0; i < xs.length; i++) {
                ypred[i] = n.apply(xs[i])[0];
            }

            // loss = sum((yout - ygt)**2)
            Value loss = value(0.0);
            for (int i = 0; i < ys.length; i++) {
                loss = loss.add(ypred[i].add(value(-ys[i])).pow(2));
            }

            // backward pass
            n.zeroGrad();
            loss.backward();

            // update
            for (Value p : n.parameters()) {
                p.setData(p.getData() + (-0.1) * p.getGrad());
            }

            System.out.printf("step %2d  loss=%.6f%n", k, loss.getData());
        }

        System.out.println();
        System.out.println("Predictions after training:");
        for (int i = 0; i < xs.length; i++) {
            double pred = n.apply(xs[i])[0].getData();
            System.out.printf("  input %d -> predicted=%.4f, target=%.1f%n", i, pred, ys[i]);
        }
    }

    public static void valueDemo() {
        System.out.println("=== Value Demo: Automatic Differentiation ===");
        System.out.println("Building a simple expression graph: f = (a * b) * (a + b)");
        System.out.println();

        Value a = new Value(-2.);
        Value b = new Value(3.);
        System.out.println("Inputs: a = " + a.getData() + ", b = " + b.getData());

        Value d = a.mul(b);
        System.out.println("d = a * b = " + d.getData());

        Value e = a.add(b);
        System.out.println("e = a + b = " + e.getData());

        Value f = d.mul(e);
        System.out.println("f = d * e = " + f.getData());

        System.out.println();
        System.out.println("Running backpropagation from f...");
        f.backward();
        System.out.println("a: " + a + "  (df/da = " + a.getGrad() + ")");
        System.out.println("b: " + b + "  (df/db = " + b.getGrad() + ")");
    }

    /**
     * Demo ported from micrograd demo.ipynb. Trains an MLP on a make_moons-like dataset using SVM max-margin (hinge) loss with
     * L2 regularization and a decaying learning rate.
     *
     * @see <a href="https://github.com/karpathy/micrograd/blob/master/demo.ipynb">demo.ipynb</a>
     */
    public static void demo() {
        System.out.println("=== Moon Demo: Binary Classification (from micrograd demo.ipynb) ===");
        System.out.println("Generating a 'make_moons' dataset: 100 samples of 2 interleaving half-circles.");
        System.out.println("Each sample has 2 features (x, y coordinates) and a label of -1 or 1.");
        System.out.println();

        Random rng = new Random(1337);

        // generate make_moons dataset (n_samples=100, noise=0.1)
        int nSamples = 100;
        double noise = 0.1;
        double[][] X = new double[nSamples][2];
        int[] y = new int[nSamples];
        for (int i = 0; i < nSamples; i++) {
            double angle = Math.PI * i / (nSamples / 2);
            if (i < nSamples / 2) {
                X[i][0] = Math.cos(angle) + rng.nextGaussian() * noise;
                X[i][1] = Math.sin(angle) + rng.nextGaussian() * noise;
                y[i] = -1;
            } else {
                X[i][0] = 1 - Math.cos(angle) + rng.nextGaussian() * noise;
                X[i][1] = 1 - Math.sin(angle) - 0.5 + rng.nextGaussian() * noise;
                y[i] = 1;
            }
        }

        System.out.println("Network architecture: MLP(2, [16, 16, 1])");
        System.out.println("  - 2 input features");
        System.out.println("  - 2 hidden layers with 16 neurons each (tanh activation)");
        System.out.println("  - 1 output (score for classification)");

        // model = MLP(2, [16, 16, 1])
        MLP model = new MLP(2, new int[] { 16, 16, 1 });
        System.out.println("Total trainable parameters: " + model.parametersLength());
        System.out.println();

        System.out.println("Loss: SVM (support vector machine) max-margin (hinge) loss + L2 regularization (alpha=1e-4)");
        System.out.println("  hinge(yi, si) = relu(1 - yi * si)");
        System.out.println("  total_loss = mean(hinge losses) + alpha * sum(p^2)");
        System.out.println("Optimizer: SGD (stochastic gradient descent) with decaying learning rate (1.0 -> 0.1 over 100 steps)");
        System.out.println();

        System.out.println("Training for 100 steps...");

        // training loop
        for (int k = 0; k < 100; k++) {

            // forward pass - compute scores for all inputs
            Value[] scores = new Value[nSamples];
            for (int i = 0; i < nSamples; i++) {
                Value[] input = { value(X[i][0]), value(X[i][1]) };
                scores[i] = model.apply(input)[0];
            }

            // svm "max-margin" loss: sum of relu(1 + -yi*scorei)
            Value dataLoss = value(0.0);
            for (int i = 0; i < nSamples; i++) {
                // (1 + -yi * scorei).relu()
                dataLoss = dataLoss.add(value(1.0).add(value(-y[i]).mul(scores[i])).relu());
            }
            dataLoss = dataLoss.mul(value(1.0 / nSamples));

            // L2 regularization
            double alpha = 1e-4;
            Value regLoss = value(0.0);
            for (Value p : model.parameters()) {
                regLoss = regLoss.add(p.mul(p));
            }
            regLoss = regLoss.mul(value(alpha));

            Value totalLoss = dataLoss.add(regLoss);

            // accuracy
            int correct = 0;
            for (int i = 0; i < nSamples; i++) {
                if ((y[i] > 0) == (scores[i].getData() > 0)) {
                    correct++;
                }
            }
            double accuracy = (double) correct / nSamples;

            // backward pass
            model.zeroGrad();
            totalLoss.backward();

            // update (SGD with decaying learning rate)
            double learningRate = 1.0 - 0.9 * k / 100;
            for (Value p : model.parameters()) {
                p.setData(p.getData() - learningRate * p.getGrad());
            }

            if (k % 10 == 0) {
                System.out.printf("  step %3d  loss=%.4f  accuracy=%3.0f%%  lr=%.3f%n",
                        k, totalLoss.getData(), accuracy * 100, learningRate);
            }
        }

        System.out.println();
        System.out.println("Training complete.");
    }
}

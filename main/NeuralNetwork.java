import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class NeuralNetwork implements Serializable {
    Layer[] layers;
    int[] structure;

    public double lambda = 0.1;

    public NeuralNetwork(int inputNeurons, int[] hiddenNeurons, int outputNeurons, double rate) {
        this.lambda = rate;
        
        this.layers = new Layer[hiddenNeurons.length + 2];

        this.structure = new int[hiddenNeurons.length + 2];
        this.structure[0] = inputNeurons;
        this.structure[this.structure.length - 1] = outputNeurons;
        for (int i = 0; i < hiddenNeurons.length; i++)
            this.structure[i + 1] = hiddenNeurons[i];

        for (int l = 0; l < this.structure.length - 1; l++) {
            int cl_size = this.structure[l];
            int nl_size = this.structure[l + 1];

            double[] neurons = new double[cl_size];
            double[][] weights = new double[cl_size][nl_size];
            double[] biases = new double[nl_size];

            for (int n = 0; n < cl_size; n++) {
                neurons[n] = 0;
            }

            double limit = Math.sqrt(6.0 / (inputNeurons + outputNeurons));
            for (int clw = 0; clw < cl_size; clw++) {
                for (int nlw = 0; nlw < nl_size; nlw++) {
                    weights[clw][nlw] = Math.random() * 2 * limit - limit;

                    if (clw == 0) {
                        biases[nlw] = Math.random() * 2 * limit - limit;
                    }
                }
            }

            this.layers[l] = new Layer(neurons, weights, biases);
        }

        double[] neurons = new double[outputNeurons];
        for (int n = 0; n < outputNeurons; n++) {
            neurons[n] = 0;
        }

        this.layers[this.layers.length - 1] = new Layer(neurons);
    }

    public double[] feedForward(double[] input) {
        this.layers[0].neurons = input;

        for (int l = 0; l < this.structure.length - 1; l++) {
            Layer currLayer = this.layers[l];
            Layer nextLayer = this.layers[l + 1];

            for (int nl = 0; nl < nextLayer.neurons.length; nl++) {
                nextLayer.neurons[nl] = 0;

                for (int cl = 0; cl < currLayer.neurons.length; cl++) {
                    double weight = currLayer.weights[cl][nl];
                    double neuron = currLayer.neurons[cl];

                    nextLayer.neurons[nl] += neuron * weight;
                }

                nextLayer.neurons[nl] += currLayer.biases[nl];
                this.layers[l + 1].neurons[nl] = activation(nextLayer.neurons[nl]);
            }
        }

        double[] output = new double[this.structure[this.structure.length - 1]];

        output = this.layers[this.structure.length - 1].neurons;

        return output;
    }

    public void backPropagation(double[] error) {
        double[] gradients = new double[this.structure[this.structure.length - 1]];

        for (int l = this.structure.length - 1; l >= 1; l--) {
            Layer currentLayer = this.layers[l];
            Layer nextLayer = this.layers[l - 1];

            if (l == this.structure.length - 1) {
                for (int cl = 0; cl < currentLayer.neurons.length; cl++) {
                    gradients[cl] = error[cl] * derivative(currentLayer.neurons[cl]);
                }
            }

            for (int cl = 0; cl < currentLayer.neurons.length; cl++) {
                for (int nl = 0; nl < nextLayer.neurons.length; nl++) {
                    double gradient = gradients[cl];
                    double neuron = nextLayer.neurons[nl];

                    this.layers[l - 1].weights[nl][cl] -= lambda * gradient * neuron;
                }
            }

            double[] newGradients = new double[this.structure[l - 1]];
            for (int nl = 0; nl < nextLayer.neurons.length; nl++) {
                newGradients[nl] = 0;

                for (int cl = 0; cl < currentLayer.neurons.length; cl++) {
                    double weight = nextLayer.weights[nl][cl];
                    double neuron = nextLayer.neurons[nl];
                    double gradient = gradients[cl];

                    newGradients[nl] += lambda * weight * gradient * derivative(neuron);
                    this.layers[l - 1].biases[cl] += newGradients[nl];
                }
            }

            gradients = newGradients;
        }
    }

    public double activation(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    public double derivative(double x) {
        return 0.5 * (1 + x) * (1 - x);
    }

    public void saveNeuralNetwork(String filename) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(filename);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.close();
    }

    public static NeuralNetwork loadNeuralNetwork(String filename) throws IOException {
        try (FileInputStream fileInputStream = new FileInputStream(filename);
                ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            return (NeuralNetwork) objectInputStream.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Не вдалось завантажити модель: " + e.getMessage(), e);
        }
    }
}

class Layer implements Serializable {
    public double[] neurons;
    public double[][] weights;
    public double[] biases;

    public Layer(double[] n) {
        this.neurons = n;
    }

    public Layer(double[] n, double[][] w, double[] b) {
        this.neurons = n;
        this.weights = w;
        this.biases = b;
    }
}
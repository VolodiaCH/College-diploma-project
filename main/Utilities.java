public class Utilities {
    public static int getIndexOfLargest(double[] array) {
        if (array == null || array.length == 0)
            return -1;

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest])
                largest = i;
        }
        return largest;
    }

    public static double[] softmax(double[] vector) {
        double sum = 0.0;

        for (double value : vector) {
            sum += Math.exp(value);
        }

        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = Math.exp(vector[i]) / sum;
        }
        
        return result;
    }
}

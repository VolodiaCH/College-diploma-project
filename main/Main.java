import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {
    private static final int IMAGE_SIZE = 28;
    private static final int PIXEL_SIZE = 20;

    private static final int batchSize = 1000;

    private JFrame frame;
    private JPanel canvas;
    private JButton computeButton, clearButton, saveProgressButton;

    private double[][] pixels = new double[IMAGE_SIZE][IMAGE_SIZE];
    public NeuralNetwork neuralNetwork;

    private boolean isLeftMousePressed = false;
    private boolean isRightMousePressed = false;

    public Main(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;

        frame = new JFrame("Нейромережа");
        frame.setResizable(false);
        frame.setAlwaysOnTop(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());

        canvas = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                drawImage(g);
            }
        };

        canvas.setPreferredSize(new Dimension(IMAGE_SIZE * PIXEL_SIZE, IMAGE_SIZE * PIXEL_SIZE));
        canvas.addMouseListener(new DrawingMouseListener());
        canvas.addMouseMotionListener(new DrawingMouseMotionListener());

        computeButton = new JButton("Аналіз");
        computeButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                analyzeImage();
            }
        });

        clearButton = new JButton("Очистити полотно");
        clearButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                clearCanvas();
            }
        });

        saveProgressButton = new JButton("Зберегти поточну модель");
        saveProgressButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                saveProgress(neuralNetwork);
            }
        });

        JPanel learningButtonsPanel = new JPanel();
        for (int i = 0; i <= 9; i++) {
            int finalI = i;
            JButton button = new JButton(String.valueOf(finalI));
            button.addActionListener(e -> learnDigit(finalI));
            learningButtonsPanel.add(button);
        }

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(saveProgressButton);
        buttonPanel.add(computeButton);
        buttonPanel.add(clearButton);

        frame.add(canvas, BorderLayout.CENTER);
        frame.add(learningButtonsPanel, BorderLayout.NORTH);
        frame.add(buttonPanel, BorderLayout.SOUTH);
        frame.pack();
        frame.setVisible(true);
    }

    private void drawImage(Graphics g) {
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int pixelValue = (int) (pixels[y][x] * 255);
                g.setColor(new Color(pixelValue, pixelValue, pixelValue));
                g.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
            }
        }
    }

    private void analyzeImage() {
        double[] flatPixels = new double[IMAGE_SIZE * IMAGE_SIZE];
        int index = 0;
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                flatPixels[index++] = pixels[y][x];
            }
        }

        double[] result = Utilities.softmax(neuralNetwork.feedForward(flatPixels));

        System.out.println("Результат аналізу:");
        for (int i = 0; i < result.length; i++) {
            System.out.println("Число " + i + ": " + Math.round(result[i]*100) + "%");
        }

        System.out.println("Результат: " + Utilities.getIndexOfLargest(result));
    }

    private void learnDigit(int digit) {
        double[] flatPixels = new double[IMAGE_SIZE * IMAGE_SIZE];
        int index = 0;
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                flatPixels[index++] = pixels[y][x];
            }
        }

        double[] result = Utilities.softmax(neuralNetwork.feedForward(flatPixels));
        double[] errors = new double[10];
        for (int i = 0; i < 10; i++) {
            if (i == digit)
                errors[i] = result[i] - 1;
            else
                errors[i] = result[i];
        }

        neuralNetwork.backPropagation(errors);

        clearCanvas();
    }

    private void clearCanvas() {
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                pixels[y][x] = 0.0;
            }
        }
        canvas.repaint();
    }

    public void saveProgress(NeuralNetwork NN) {
        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.pack();
        f.setAlwaysOnTop(true);
        f.setVisible(false);

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int result = fileChooser.showSaveDialog(f);
    
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            try {
                NN.saveNeuralNetwork(selectedFile.getAbsolutePath());
                System.out.println("Модель успішно збережена!");
            } catch (IOException e) {
                System.err.println("Не вдалось зберегти модель: " + e.getMessage());
            }
        }
    }    

    public static void loadProgress() {
        JFrame f = new JFrame();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.pack();
        f.setAlwaysOnTop(true);
        f.setVisible(false);
    
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        int result = fileChooser.showOpenDialog(f);

        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            try {
                NeuralNetwork NN = NeuralNetwork.loadNeuralNetwork(selectedFile.getAbsolutePath());
                
                SwingUtilities.invokeLater(() -> new Main(NN));
            } catch (IOException e) {
                System.err.println("Не вдалось завантажити модель: " + e.getMessage());
            }
        }
    }

    public static NeuralNetwork learnModel(int epochs, int[] hiddenNeurons, double rate) {
        NeuralNetwork NN = new NeuralNetwork(IMAGE_SIZE*IMAGE_SIZE, hiddenNeurons, 10, rate);
        List<ImageData> trainData = new ArrayList<>();

        System.out.println("\nЗчитування зображень чисел...");
        trainData.addAll(ImageReader.readImagesFromPath("train", 60000));
        System.out.println("\nФайли оброблено успішно!\n");

        int counter = 0;
        int e_counter = 1;
        int correct = 0;
        int wrong = 0;

        System.out.println("Початок навчання моделі...");

        for (int j = 0; j < epochs*batchSize; j++) {
            int randomImageIdx = (int) (Math.random() * 60000);
            ImageData image = trainData.get(randomImageIdx);

            double[] pixels = new double[28 * 28];

            int p = 0;
            for (double[] y : image.pixels) {
                for (double x : y) {
                    pixels[p] = x;
                    p++;
                }
            }

            double[] errors = new double[10];
            double[] result = Utilities.softmax(NN.feedForward(pixels));

            for (int i = 0; i < 10; i++) {
                if (i == image.number)
                    errors[i] = result[i] - 1;
                else
                    errors[i] = result[i];
            }

            if (Utilities.getIndexOfLargest(result) == image.number) correct++;
            else wrong++;

            if ((counter + 1) % batchSize == 0 && counter != 0) {
                System.out.println("Епоха " + e_counter + ": Не правильних: " + wrong + "; Правильних: " + correct + ";");
                wrong = 0;
                correct = 0;
                e_counter++;
            }

            NN.backPropagation(errors);

            counter++;
        }

        return NN;
    }

    public static void testModel(NeuralNetwork NN) {
        System.out.println("\nЗчитування зображень чисел...");
        List<ImageData> testData = new ArrayList<>();
        testData.addAll(ImageReader.readImagesFromPath("test", 10000));
        System.out.print("\n");

        int counter = 0;
        int correct = 0;
        int wrong = 0;

        for (ImageData image : testData) {
            double[] pixels = new double[28 * 28];

            int p = 0;
            for (double[] y : image.pixels) {
                for (double x : y) {
                    pixels[p] = x;
                    p++;
                }
            }

            double[] result = Utilities.softmax(NN.feedForward(pixels));

            if (Utilities.getIndexOfLargest(result) == image.number) correct++;
            else wrong++;

            if ((counter + 1) % 10000 == 0 && counter != 0) {
                System.out.println((counter + 1) + "/10000 опрацьовано. Не правильних: " + wrong + "; Правильних: " + correct);
            }
            
            counter++;
        }

        System.out.printf("\nТочність: %d%%\n", (int)((double)correct/10000.0*100.0));
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        System.out.println("Оберіть варіант:");
        System.out.println("1. Нова модель\n2. Завантажити модель\n3. Завантажити пусту модель\n0. Вийти\n");
        System.out.print("Варіант роботи програми: ");
        double choice = in.nextInt();

        if (choice == 1) {
            System.out.printf("\nКількість епох (по %d зображень): ", batchSize);
            int epochs = in.nextInt();

            double rate = 0.1;
            System.out.print("\nЛямбда: ");
            if (in.hasNextDouble()) 
                rate = in.nextDouble();

            System.out.print("Кількість прихованих шарів: ");
            int hlc = 0;
            while (in.hasNextInt()) {
                hlc = in.nextInt();
                if (hlc >= 1 && hlc <= 5) {
                    break;
                } else {
                    System.out.println("Введіть число в межах [1, 5]:");
                }
            }

            int[] hiddenNeurons = new int[hlc];

            for (int i = 0; i < hlc; i++) {
                System.out.print("Шар "+(i+1)+": ");
                while (in.hasNextInt()) {
                    int neurons = in.nextInt();
                    hiddenNeurons[i] = neurons;
                    if (neurons >= 1) {
                        break;
                    } else {
                        System.out.println("Введіть додатнє число:");
                    }
                }
            }

            in.close();

            NeuralNetwork NN = learnModel(epochs, hiddenNeurons, rate);

            System.out.println("Модель успішно навчена! Тестування...");
            testModel(NN);

            SwingUtilities.invokeLater(() -> new Main(NN));
        } else if (choice == 2) {
            in.close();
            loadProgress();
        } else if (choice == 3) {
            in.close();
            int[] hiddenNeurons = {32};
            NeuralNetwork NN = new NeuralNetwork(IMAGE_SIZE*IMAGE_SIZE, hiddenNeurons, 10, 0.1);
            SwingUtilities.invokeLater(() -> new Main(NN));
        } else {
            in.close();
            System.out.println("\nВихід з програми...");
            return;
        }
    }

    private class DrawingMouseMotionListener extends MouseAdapter {
        @Override
        public void mouseDragged(MouseEvent e) {
            int x = e.getX() / PIXEL_SIZE;
            int y = e.getY() / PIXEL_SIZE;

            if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
                if (isLeftMousePressed) {
                    pixels[y][x] = 1.0;

                    if (y < IMAGE_SIZE-1 && pixels[y+1][x] == 0)
                        pixels[y+1][x] = 0.3;
                    if (y > 0 && pixels[y-1][x] == 0)
                        pixels[y-1][x] = 0.3;
                    if (x < IMAGE_SIZE-1 && pixels[y][x+1] == 0)
                        pixels[y][x+1] = 0.3;
                    if (x > 0 && pixels[y][x-1] == 0)
                        pixels[y][x-1] = 0.3;

                    canvas.repaint();
                } else if (isRightMousePressed) {
                    pixels[y][x] = 0.0;

                    if (y < IMAGE_SIZE-1 && pixels[y+1][x] > 0.3)
                        pixels[y+1][x] -= 0.3;
                    if (y > 0 && pixels[y-1][x] > 0.3)
                        pixels[y-1][x] -= 0.3;
                    if (x < IMAGE_SIZE-1 && pixels[y][x+1] > 0.3)
                        pixels[y][x+1] -= 0.3;
                    if (x > 0 && pixels[y][x-1] > 0.3)
                        pixels[y][x-1] -= 0.3;

                    canvas.repaint();
                }
            }
        }
    }

    private class DrawingMouseListener extends MouseAdapter {
        @Override
        public void mousePressed(MouseEvent e) {
            int x = e.getX() / PIXEL_SIZE;
            int y = e.getY() / PIXEL_SIZE;

            if (x >= 0 && x < IMAGE_SIZE && y >= 0 && y < IMAGE_SIZE) {
                if (SwingUtilities.isLeftMouseButton(e)) {
                    pixels[y][x] = 1.0;
                    isLeftMousePressed = true;
                    canvas.repaint();
                } else if (SwingUtilities.isRightMouseButton(e)) {
                    pixels[y][x] = 0.0;
                    isRightMousePressed = true;
                    canvas.repaint();
                }
            }
        }

        @Override
        public void mouseReleased(MouseEvent e) {
            if (SwingUtilities.isLeftMouseButton(e)) {
                isLeftMousePressed = false;
                isRightMousePressed = false;
            }
        }
    }
}

class ImageData {
    public double[][] pixels;
    public int number;

    public ImageData(double[][] pixels, int number) {
        this.pixels = pixels;
        this.number = number;
    }
}
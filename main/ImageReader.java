import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import java.awt.image.BufferedImage;

public class ImageReader {
    private static double[][] parseImageToPixels(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] pixels = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                double grayscaleValue = ((rgb >> 16) & 0xFF) / 255.0;
                pixels[y][x] = grayscaleValue;
            }
        }

        return pixels;
    }

    private static int extractNumberFromFileName(String fileName) {
        if (fileName.length() >= 11) {
            char numChar = fileName.charAt(10);
            if (Character.isDigit(numChar)) {
                return Character.getNumericValue(numChar);
            }
        }
        return -1;
    }

    public static List<ImageData> readImagesFromPath(String path, int numberOfImages) {
        List<ImageData> imageDataList = new ArrayList<>();

        File directory = new File("../" + path);

        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();

            int processedFiles = 0;

            for (File file : files) {
                if (file.isFile() && file.getName().endsWith(".png")) {
                    try {
                        BufferedImage image = ImageIO.read(file);
                        double[][] pixels = parseImageToPixels(image);
                        int number = extractNumberFromFileName(file.getName());

                        ImageData imageData = new ImageData(pixels, number);
                        imageDataList.add(imageData);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    processedFiles++;
                    System.out.printf("\rОбробка зображень... (%d/%d)", processedFiles, numberOfImages);
                }
            }
        }

        return imageDataList;
    }
}

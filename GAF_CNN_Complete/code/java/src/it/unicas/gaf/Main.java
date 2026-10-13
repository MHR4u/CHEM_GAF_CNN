package it.unicas.gaf;

import java.io.File;

//import static it.unicas.gaf.GramImage.GASF;

public class Main {

    // GASF/GADF




    public static void main(String[] args) throws Exception {


        // CREATE A png image from byte
        //
        //        byte[] aByteArray = {
        //                //  R,      G,          B
        //                (byte)0x00,(byte)0x00,(byte)0xff,(byte)0x00,(byte)0xff,(byte)0x00,  // I row
        //                (byte)0x00,(byte)0x00,(byte)0x00,(byte)0xff,(byte)0x00,(byte)0x00}; // II row
        //        int width = 2;
        //        int height = 2;
        //
        //        DataBuffer buffer = new DataBufferByte(aByteArray, aByteArray.length);
        //
        //        //3 bytes per pixel: red, green, blue
        //        WritableRaster raster = Raster.createInterleavedRaster(buffer, width, height, 3 * width, 3, new int[] {0, 1, 2}, (Point)null);
        //        ColorModel cm = new ComponentColorModel(ColorModel.getRGBdefault().getColorSpace(), false, true, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
        //        BufferedImage image = new BufferedImage(cm, raster, true, null);
        //        ImageIO.write(image, "png", new File("image.png"));



        // https://arxiv.org/pdf/1506.00327.pdf
        // GAF


        // Input Tor Vergata **********************
        // String substances[] = new String[]{"Benzochinone", "Idrochinone", "FerricianuroDiPotassio"};
        // String startBaseDir = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TorVergata/";
        // String[] sensors = new String[]{"SWCNT_SPEs", "MWCNT_SPEs", "Bare_SPEs"};

        // Input Tor Vergata, esperimenti settembre 2023 **********************
        // String substances[] = new String[]{"Benzochinone", "Idrochinone"};
        // String startBaseDir = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TorVergata/settembre2023/";
        // String[] sensors = new String[]{"Misha-SPE", "MWNT-SPE"};

        // Input Tor Vergata, esperimenti ottobre 2023 **********************
        // String substances[] = new String[]{"Benzochinone", "Idrochinone", "Benzo-Idrochinone"};
        // String startBaseDir = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TorVergata/ottobre2023/";
        // String[] sensors = new String[]{"SPE"};


        String substances[] = new String[]{"Benzochinone", "Idrochinone", "Catecolo", "Ferro-Ferri", "Idrochinone-Benzochinone"};
        String startBaseDir = args[0]; // "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TorVergata/maggio2024/";
        String[] sensors = new String[]{"SPE"};


        // Tensorflow directory
        // String tensorFlowDirectory = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TransferLearning/classes/";

        // Tensorflow directory, settembre 2023
        // String tensorFlowDirectory = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TransferLearning/classes_settembre2023/";

        // Tensorflow directory, ottobre 2023
        String tensorFlowDirectory = args[1]; //"/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/TransferLearning/classes_maggio2024/";


        float trainFraction = 0.6f;
        float testFraction = 0.2f;

        int totalSavedImages = 0;

        //int[] GAF_METHODS = new int[]{GramImage.GADF}; //, GramImage.GASF};
        //System.out.println("V min\t\tV max\t\tI min\t\tI max");

        //for(int k = 0; k < GAF_METHODS.length; k++){
            //System.out.println("GAF method: " + (GAF_METHODS[k] == GASF ? "GASF" : "GADF"));

            for(int i = 0; i < substances.length; i++){
                String baseDir = startBaseDir + substances[i] + "/" + "GAFData" + "/";
                System.out.println("Substance: " + substances[i]);

                for(int j = 0; j < sensors.length; j++){
                    System.out.println("Sensor: " + sensors[j]);

                    String inputFileName = sensors[j] + "-" + substances[i] + ".csv";
                    System.out.println(inputFileName);

                    GramianImageCollector gramianImageCollector = new GramianImageCollector(baseDir, totalSavedImages, inputFileName, substances[i], sensors[j]);//, GAF_METHODS[k]);

                    System.out.println(gramianImageCollector.getMaxMinV_MaxMinI()[GramImage.MIN_V_INDEX] +
                        ", " + gramianImageCollector.getMaxMinV_MaxMinI()[GramImage.MAX_V_INDEX] + "\t" +
                        ", " + gramianImageCollector.getMaxMinV_MaxMinI()[GramImage.MIN_I_INDEX] + "\t" +
                        ", " + gramianImageCollector.getMaxMinV_MaxMinI()[GramImage.MAX_I_INDEX]);

                    int counter = 1;
                    int totalNumberOfImages = 0;
                    for(int l = 0; l < gramianImageCollector.gramImages.length; l++){
                        for(int m = 0; m < gramianImageCollector.gramImages[l].images.size(); m++) {
                            totalNumberOfImages++;
                        }
                    }
                    for(int l = 0; l < gramianImageCollector.gramImages.length; l++){
                        for(int m = 0; m < gramianImageCollector.gramImages[l].images.size(); m++){

                            String outputFile = tensorFlowDirectory;
                            float randomVal = (float)Math.random();

                            if (randomVal < trainFraction){
                                outputFile += "Train/";
                            } else if (randomVal < (trainFraction + testFraction)){
                                outputFile += "Test/";
                            } else {
                                outputFile += "Val/";
                            }
                            outputFile += substances[i] + "/";
                            new File(outputFile).mkdirs();

                            outputFile += gramianImageCollector.gramImages[l].fileNames.get(m); //.replace(".png", "_" + (totalSavedImages + 1)) + ".png";

                            //System.out.println(outputFile);
                            if (new File(outputFile).exists()){
                                System.out.println(outputFile + ". File already exists!");
                                //throw new Exception("File already exists!");
                            }
                            gramianImageCollector.generateImage(
                                    gramianImageCollector.gramImages[l].images.get(m),
                                    outputFile,
                                    gramianImageCollector.gramImages[l].imageSize);
                            counter ++;
                            totalSavedImages++;
                        }
                    }

                }
            }
        //}
        System.out.println("Total saved images: " + totalSavedImages);
    }
}

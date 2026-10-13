package it.unicas.gaf;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static it.unicas.gaf.GramImage.*;

public class GramianImageCollector {

    public static String COMMA_DELIMITER = ";";

    public static String VOLTAGE_LABEL = "V";
    public GramImage gramImages[];
    float[] maxMinV_MaxMinI = new float[4];

    public static boolean USE_TRAINING_MAX = true;


    public GramianImageCollector(String baseDir, int idCollection, String fileName, String substance, String sensor) throws IOException { //}, int gasf_type) throws IOException {


        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(baseDir + fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(COMMA_DELIMITER);
                records.add(Arrays.asList(values));
            }
        }

        if (USE_TRAINING_MAX){
            maxMinV_MaxMinI[MAX_V_INDEX] = Float.MIN_VALUE;
            maxMinV_MaxMinI[MIN_V_INDEX] = Float.MAX_VALUE;
            maxMinV_MaxMinI[MAX_I_INDEX] = Float.MIN_VALUE;
            maxMinV_MaxMinI[MIN_I_INDEX] = Float.MAX_VALUE;
        } else {
            maxMinV_MaxMinI = getPredefinedMaxMin();
        }

        int concentrationCounter = 1;
        int imagesForSingleConcentration = 0;
        boolean flag = false;
        List<String> secondLine = records.get(1);
        // Si sta supponendo che la seconda riga cominci con V
        // che ogni gruppo di colonne cominci con V
        // che le colonne che contengono il valore di corrente, non contengano V
        for(int i = 1; i < records.get(1).size(); i++){
            if (!secondLine.get(i).equals(VOLTAGE_LABEL) && !flag){
                imagesForSingleConcentration++;
            } else if (secondLine.get(i).equals(VOLTAGE_LABEL)){
                flag = true;
                concentrationCounter++;
            }
        }

        gramImages = new GramImage[concentrationCounter];

        for(int i = 0; i < concentrationCounter; i++){
            gramImages[i] = new GramImage(records, fileName,
                    i * (imagesForSingleConcentration + 1),
                    i * (imagesForSingleConcentration + 1) + imagesForSingleConcentration,
                    substance);

            // Evaluate max of max's and min of min's
            if (gramImages[i].getMaxMinV_MaxMinI()[MAX_V_INDEX] > maxMinV_MaxMinI[MAX_V_INDEX]){
                maxMinV_MaxMinI[MAX_V_INDEX] = gramImages[i].getMaxMinV_MaxMinI()[MAX_V_INDEX];
            }
            if (gramImages[i].getMaxMinV_MaxMinI()[MIN_V_INDEX] < maxMinV_MaxMinI[MIN_V_INDEX]){
                maxMinV_MaxMinI[MIN_V_INDEX] = gramImages[i].getMaxMinV_MaxMinI()[MIN_V_INDEX];
            }
            if (gramImages[i].getMaxMinV_MaxMinI()[MAX_I_INDEX] > maxMinV_MaxMinI[MAX_I_INDEX]){
                maxMinV_MaxMinI[MAX_I_INDEX] = gramImages[i].getMaxMinV_MaxMinI()[MAX_I_INDEX];
            }
            if (gramImages[i].getMaxMinV_MaxMinI()[MIN_I_INDEX] < maxMinV_MaxMinI[MIN_I_INDEX]){
                maxMinV_MaxMinI[MIN_I_INDEX] = gramImages[i].getMaxMinV_MaxMinI()[MIN_I_INDEX];
            }
        }

        for(int i = 0; i < concentrationCounter; i++){
            gramImages[i].normalizeAndGenerateImage(getMaxMinV_MaxMinI());
            for(int j = 0; j < gramImages[i].images.size(); j++){

                File outputDir = new File(baseDir +
                        //(gasf_type == GASF ? "GASF" : "GADF") +
                        //"/" +
                        sensor +
                        "/");
                outputDir.mkdirs();
                String completeFileName = outputDir + "/" + gramImages[i].fileNames.get(j);
                System.out.println(completeFileName);
                //if (new File(completeFileName).exists()){
                //    throw new IOException("File already exists in normal generation!");
                //}
                generateImage(gramImages[i].images.get(j), completeFileName, gramImages[i].imageSize);
            }
        }
    }

    public void generateImage(byte[] aByteArray, String imgFileName, int imgSize) throws IOException{
        int width = imgSize;
        int height = imgSize;

        DataBuffer buffer = new DataBufferByte(aByteArray, aByteArray.length);

        //3 bytes per pixel: red, green, blue
        WritableRaster raster = Raster.createInterleavedRaster(buffer, width, height, 3 * width, 3, new int[] {0, 1, 2}, (Point)null);
        ColorModel cm = new ComponentColorModel(ColorModel.getRGBdefault().getColorSpace(), false, true, Transparency.OPAQUE, DataBuffer.TYPE_BYTE);
        BufferedImage image = new BufferedImage(cm, raster, true, null);

        ImageIO.write(image, "png", new File(imgFileName));

    }

    float[] getMaxMinV_MaxMinI(){
        return maxMinV_MaxMinI;
    }

    public static float[] getPredefinedMaxMin(){
        return new float[]{1.50479f, -1.50003f, 3448.39f, -2537.1f};
    }

    //    -1.50003, 1.19518	, -1055.85	, 972.808
    //            -1.50003, 1.19518	, -1055.85	, 972.808
    //            -1.50003, 1.19518	, -952.392	, 822.568
    //            -1.50003, 1.19518	, -952.392	, 822.568
    //            -1.50003, 0.995538	, -388.418	, 228.917
    //            -1.50003, 0.995538	, -388.418	, 228.917
    //            -1.40005, 1.50479	, -2537.1	, 3448.39
    //            -1.40005, 1.50479	, -2537.1	, 3448.39
    //            -1.40005, 1.50479	, -2417.44	, 3313.8
    //            -1.40005, 1.50479	, -2417.44	, 3313.8
    //            -1.40005, 1.50479	, -1362.74	, 2642.39
    //            -1.40005, 1.50479	, -1362.74	, 2642.39

}

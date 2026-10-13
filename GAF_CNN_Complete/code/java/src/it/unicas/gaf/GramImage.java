package it.unicas.gaf;

import java.util.ArrayList;
import java.util.List;

public class GramImage {
    private String substance;
    private String concentration;
    //private int numOfExperiments;

    private float[] rawVoltage;
    private float[][] rawCurrent;
    float[] maxMinV_MaxMinI = new float[4];

    public ArrayList<byte[]> images = new ArrayList<>();
    public ArrayList<String> fileNames = new ArrayList<>();

    public static int MAX_V_INDEX = 0;
    public static int MIN_V_INDEX = 1;
    public static int MAX_I_INDEX = 2;
    public static int MIN_I_INDEX = 3;

    //public static int GADF = 0;
    //public static int GASF = 1;
    //private int gaf_type;
    //private int GAF_TYPE = GramImage.GASF;

    public int imageSize;
    private String fileName = "";

    public GramImage(List<List<String>> data, String fileName, int startColumn, int endColumn, String substance){//}, int gaf_type){

        //this.gaf_type = gaf_type;
        this.fileName = fileName;

        concentration = data.get(0).get(startColumn);
        imageSize = data.size() - 2; // Do not consider first two rows
        rawVoltage = new float[imageSize];
        rawCurrent = new float[imageSize][];

        for(int i = 0; i < (endColumn - startColumn); i++){
            images.add(new byte[imageSize * imageSize * 3]);
        }
        this.substance = substance;

        if (GramianImageCollector.USE_TRAINING_MAX){
            maxMinV_MaxMinI[MAX_V_INDEX] = Float.MIN_VALUE;
            maxMinV_MaxMinI[MIN_V_INDEX] = Float.MAX_VALUE;
            maxMinV_MaxMinI[MAX_I_INDEX] = Float.MIN_VALUE;
            maxMinV_MaxMinI[MIN_I_INDEX] = Float.MAX_VALUE;
        } else {
            maxMinV_MaxMinI = GramianImageCollector.getPredefinedMaxMin();
        }


        for(int i = 2; i < imageSize + 2; i++){
            rawVoltage[i - 2] = Float.parseFloat(data.get(i).get(startColumn));
            if (rawVoltage[i - 2] > maxMinV_MaxMinI[MAX_V_INDEX]){
                maxMinV_MaxMinI[MAX_V_INDEX] = rawVoltage[i - 2];
            } else if (rawVoltage[i - 2] < maxMinV_MaxMinI[MIN_V_INDEX]){
                maxMinV_MaxMinI[MIN_V_INDEX] = rawVoltage[i - 2];
            }
            rawCurrent[i - 2] = new float[endColumn - startColumn];
            for(int j = startColumn + 1; j <= endColumn; j++){
                rawCurrent[i - 2][j - startColumn - 1] = Float.parseFloat(data.get(i).get(j));
                if (rawCurrent[i - 2][j - startColumn - 1] > maxMinV_MaxMinI[MAX_I_INDEX]){
                    maxMinV_MaxMinI[MAX_I_INDEX] = rawCurrent[i - 2][j - startColumn - 1];
                } else if (rawCurrent[i - 2][j - startColumn - 1] < maxMinV_MaxMinI[MIN_I_INDEX]){
                    maxMinV_MaxMinI[MIN_I_INDEX] = rawCurrent[i - 2][j - startColumn - 1];
                }
            }
        }


    }


    public String getSubstance(){
        return substance;
    }

    public String getConcentration(){
        return concentration;
    }


    float[] getMaxMinV_MaxMinI(){
        return maxMinV_MaxMinI;
    }

    public void normalizeAndGenerateImage(float[] maxMinV_MaxMinI){
        for(int i = 0; i < rawVoltage.length; i++){
            rawVoltage[i] = ((rawVoltage[i] - maxMinV_MaxMinI[MAX_V_INDEX]) + (rawVoltage[i] - maxMinV_MaxMinI[MIN_V_INDEX]))/(maxMinV_MaxMinI[MAX_V_INDEX] - maxMinV_MaxMinI[MIN_V_INDEX]);
            rawVoltage[i] = (float)Math.acos(rawVoltage[i]); // TODO: Verificare!!!
            for(int j = 0; j < rawCurrent[i].length; j++){
                rawCurrent[i][j] = ((rawCurrent[i][j] - maxMinV_MaxMinI[MAX_I_INDEX]) + (rawCurrent[i][j] - maxMinV_MaxMinI[MIN_I_INDEX]))/(maxMinV_MaxMinI[MAX_I_INDEX] - maxMinV_MaxMinI[MIN_I_INDEX]);
                rawCurrent[i][j] = (float)Math.acos(rawCurrent[i][j]);
            }
        }

        for(int k = 0; k < images.size(); k++){
            byte[] image = images.get(k);
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    float r = 0;
                    float g = 0;
                    float b = 0;
                    //if(GAF_TYPE == GASF){
                          r = (float)Math.cos(rawVoltage[i] + rawVoltage[j]) + 1;
                          g = (float)Math.cos(rawCurrent[i][k] + rawCurrent[j][k]) + 1;
                    //} else {
                    //    r = (float)Math.sin(rawVoltage[i] - rawVoltage[j]) + 1;
                        b = (float)Math.sin(rawCurrent[i][k] - rawCurrent[j][k]) + 1;
                    //}
                    if (r < 0){
                        System.err.println("Negative value for color R");
                    }
                    image[i * imageSize * 3 + j * 3]        = (byte)Math.round((255 * r/2));    // Red
                    image[i * imageSize * 3 + j * 3 + 1]    = (byte)Math.round((255 * g/2));    // Green
                    image[i * imageSize * 3 + j * 3 + 2]    = (byte)Math.round((255 * b/2));                       // Blue
                }
            }
            String imgFileName = getConcentration().replace(",",".").replace(" ","");
            imgFileName += fileName.replace(".csv", "");
            //imgFileName += gaf_type == GASF ? "GASF" : "GADF";
            imgFileName += "_" + (k + 1);
            imgFileName += ".png";

            fileNames.add(imgFileName);
        }

    }




}

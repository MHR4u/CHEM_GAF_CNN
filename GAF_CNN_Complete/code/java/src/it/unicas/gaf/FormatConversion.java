package it.unicas.gaf;

import java.io.File;
import java.util.ArrayList;
import java.util.Hashtable;

public class FormatConversion {

    private Hashtable <String, ArrayList<String>> table = new Hashtable<>();
    private ArrayList<Global> listOfGlobals = new ArrayList<>();

    public FormatConversion(String mainDir) throws Exception {
        File substancesDirs[] = new File(mainDir).listFiles(pathname -> pathname.isDirectory());
        String sensorName = "PLATINUM_IDE";
        String concentration = "10mM";
        String substance = "";
        for(File item: substancesDirs){
            substance = item.getName();
            System.out.println(item.getName());
            File[] trials = item.listFiles();
            for(File trial: trials){
                System.out.println("\t" + trial.getAbsolutePath() + "/GLOBAL.csv");
                Global global = new Global(trial.getAbsolutePath() + "/GLOBAL.csv", substance, sensorName, concentration);
                listOfGlobals.add(global);
            }

        }

    }

    public void mergeAndSaveSubstances(String outputDir) throws Exception {
        int numberOfExperimentsPerSubstance = 10;
        ArrayList<Global> mergedGlobal = new ArrayList<>();
        Global toMerge = null;
        for(int i = 0; i < listOfGlobals.size(); i++){
            if (i % numberOfExperimentsPerSubstance == 0){
                toMerge = listOfGlobals.get(i);
                mergedGlobal.add(toMerge);
            } else {
                toMerge.union(listOfGlobals.get(i));
            }
        }

        for(Global item: mergedGlobal){
            item.extractCycles();
            item.saveOUtputCSVFiles(outputDir);
        }
    }


    public static void main(String[] args) throws Exception{
        String mainDir = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/May19EmanueleFranzese/dpv";
        String outputDir = "/home/mario/gdrive/Sensichips/SENSIPLUS_ResearchActivities/Papers/CyclicVoltammetry/May19EmanueleFranzese/dpv_newformat/";

        FormatConversion formatConversion = new FormatConversion(mainDir);
        formatConversion.mergeAndSaveSubstances(outputDir);
    }

}

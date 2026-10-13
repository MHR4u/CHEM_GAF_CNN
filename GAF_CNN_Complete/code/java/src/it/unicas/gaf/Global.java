package it.unicas.gaf;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static it.unicas.gaf.GramianImageCollector.COMMA_DELIMITER;

public class Global {

    private static int VOLTAGE_INDEX = 3;
    private static int CURRENT_INDEX = 4;


    public float[][] outputTable;

    private static float MULTIPLIER = 1e-3f;
    private static float MAX_V = 1.25f;
    private static float MIN_V = -1.25f;

    private List<Float[]> inputRows = new ArrayList<>();
    private List<Float[]> outputRows = new ArrayList<>();
    private String substance = "";
    private String sensorName = "";
    private String concentration = "";
    private int cycleCounter;
    public Global(String fileToLoad, String substance, String sensorName, String concentration) throws Exception {
        try (BufferedReader br = new BufferedReader(new FileReader(fileToLoad))) {
            this.substance = substance;
            this.sensorName = sensorName;
            this.concentration = concentration;
            String line;
            boolean flag = false; // Avoid first line
            while ((line = br.readLine()) != null) {
                if (!flag){
                    flag = true;
                    continue;
                }
                String[] valuesString = line.split(COMMA_DELIMITER);
                Float[] values = new Float[2]; // Only for voltage and current
                values[0] = Float.parseFloat(valuesString[VOLTAGE_INDEX])*MULTIPLIER;
                values[1] = Float.parseFloat(valuesString[CURRENT_INDEX]);
                inputRows.add(values);
            }
        }

        return;
    }

    public void union(Global global) throws Exception {
        if (!global.concentration.equals(concentration)  ||
        !global.substance.equals(substance) ||
        !global.sensorName.equals(sensorName)){
            throw new Exception("Problem during unione of two globals!");
        }
        inputRows.remove(inputRows.size() - 1);
        inputRows.addAll(global.inputRows);
    }

    public void extractCycles() throws Exception {
        // Detect voltammetry cycle
        cycleCounter = 0;
        // Ipotesi: si parte dal minimo negativo
        int previousIndex = Integer.MIN_VALUE;
        ArrayList<Integer> startingIndexes = new ArrayList<>();
        int sequenceLength = 0;
        int lastSequenceLength = Integer.MIN_VALUE;
        for(int i = 0; i < inputRows.size(); i++){
            if (inputRows.get(i)[0] == MIN_V && (i == 0 || inputRows.get(i - 1)[0] == MIN_V) && i != (inputRows.size() - 1)){
                cycleCounter++;
                //previousIndex = i;
                if (startingIndexes.size() > 0){
                    sequenceLength = i - startingIndexes.get(startingIndexes.size() - 1);
                    if (lastSequenceLength > 0 && lastSequenceLength != sequenceLength){
                        throw new Exception("Sequences of different lengths");
                    }
                    lastSequenceLength = sequenceLength;
                }
                startingIndexes.add(i);
                //System.out.println("Starting cycle at row: " + i + (lastSequenceLength > 0 ? (" with length: " + lastSequenceLength) : ""));
                //System.out.println("V: " + inputRows.get(i)[0] + ", I: " + inputRows.get(i)[1]);
            }
        }

        System.out.println("Number of cycle: " + cycleCounter);
        System.out.println("Each of length: " + sequenceLength);
        outputTable = new float[sequenceLength][cycleCounter + 1];
        for(int i = 0; i < sequenceLength; i++){
            //outputTable[i] = new float[sequenceLength];
            outputTable[i][0] = inputRows.get(i)[0];
            for(int j = 0; j < cycleCounter; j++){
                outputTable[i][j + 1] = inputRows.get(i + j * sequenceLength)[1];
            }
        }
    }


    public void saveOUtputCSVFiles(String outputDir) throws IOException {
        File dir = new File(outputDir + substance);
        dir.mkdirs();
        String fileName = sensorName + "_" + substance + ".csv";
        String firstLine = concentration + ";";
        for(int i = 0; i < cycleCounter; i++){
            firstLine += "SPE_" + (i + 1) + ";";
        }
        String secondLine = "V;";
        for(int i = 0; i < cycleCounter; i++) {
            secondLine += "µA;";
        }
        fileName = dir.getAbsolutePath() + "/" + fileName;
        if(new File(fileName).exists()){
            System.err.println("File already exists: " + fileName);
        }
        PrintWriter outputFile = new PrintWriter(fileName);
        outputFile.println(firstLine);
        outputFile.println(secondLine);
        String line = "";
        for(int i = 0; i < outputTable.length; i++){
            line = "";
            for(int j = 0; j < outputTable[i].length; j++){
                line += outputTable[i][j] + ";";
            }
            outputFile.println(line);
        }
        outputFile.close();

    }

}

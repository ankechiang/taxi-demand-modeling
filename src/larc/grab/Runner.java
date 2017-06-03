import java.io.*;
import java.util.*;

public class Runner {
    static int G;


    static Booking[] readDataWindow(double wstart, double wend, String inputPath) {
        try {
            HashMap<String, Integer> gridName2Index = new HashMap<String, Integer>();
            BufferedReader br = new BufferedReader(new FileReader(inputPath));
            String line = null;
            int nInstances = 0;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");
                // check time
                double time = Double.parseDouble(tokens[1]);
                if ((time < (60.0*wend)) && ((wstart*60.0) <= time)) {
                    nInstances++;
                }
            }
            br.close();
            Booking[] data = new Booking[nInstances];

            br = new BufferedReader(new FileReader(inputPath));
            line = null;
            nInstances = 0;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(",");

                // check time
                double time = Double.parseDouble(tokens[1]);
                if ((time < (wend*60.0)) && ((wstart*60.0) <= time)) {
                    data[nInstances] = new Booking();
                    data[nInstances].day = tokens[0];
                    data[nInstances].gridName = tokens[tokens.length - 2] + ","
                            + tokens[tokens.length - 1];
                    if (!gridName2Index.containsKey(data[nInstances].gridName))
                        gridName2Index.put(data[nInstances].gridName,
                                gridName2Index.size());
                    data[nInstances].gridIndex = gridName2Index.get(data[nInstances].gridName);
                    data[nInstances].features = new double[2];
                    for (int j = 0; j < 2; j++)
                        data[nInstances].features[j] = Double.parseDouble(tokens[j + 2]);
                    nInstances++;
                }
            }
            br.close();
            G = gridName2Index.size();
            return data;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return null;
    }


    static void runModels(String[] args) {
        try {
            String day = args[2];
            String table = args[5];
            String rootdir = args[6];
            int inr = Parameter.temporalWindow;
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    rootdir+"/grabTaxi/output/logLik" + args[0] + "_"
                            + args[1] + "_" + args[2] + "_jul_sep_d" + inr + ".csv"));
            System.out.println("read\t"+rootdir+"/grabTaxi/output/logLik" + args[0] + "_" + args[1] + "_" + args[2] + "_jul_sep_d" + inr + ".csv");
            for (int k = Integer.parseInt(args[0]); k <= Integer.parseInt(args[1]); k += inr) {
                double ggmmLoglikelihood = 0.0;
                double gmmLoglikelihood = 0.0;
                for (int w = 0; w < 24; w += inr) {
                    Booking[] data = readDataWindow(w, w + inr, rootdir+"/grabTaxi/input/sg_" + table + "_jul_sep.txt_" + day + "_b100.csv");
                    GGMM model = new GGMM(50, new Random(0));

                    System.out.println(k + "\tw(" + w + "," + (w + inr) + ")\t" + data.length);
                    model.learnSoftGGMM(k, data);
                    model.getLogLikelihood("ggmm", data);
                    ggmmLoglikelihood += model.modelLoglikelihood;
                    model.outputParameters(rootdir+"/grabTaxi/output/GGMM", "ggmm_" + day + "_jul_sep_k" + k + "_w" + (w - inr) + "_w" + w);
                }
                System.out.println("\t="+gmmLoglikelihood);
                bw.write(k * (24 / Parameter.temporalWindow) + "," + ggmmLoglikelihood + "," + gmmLoglikelihood + "\n");
                bw.flush();
            }
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }


    public static void main(String[] args) {
        System.out.println("START");
        runModels(args);
        System.out.println("DONE");
    }
}

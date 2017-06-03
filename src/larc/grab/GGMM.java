/***
 * Grid-based GMM model
 */
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashSet;
import java.util.Random;
import java.util.*;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;

public class GGMM {
    // model parameters
    private int K; // #clusters
    private int G; // #grid cells
    private double[][] pi;
    private double[] gmmPi;
    private double[][] mu;
    private double[][][] sigma;
    private int dim; // #features
    private double[][] loglikelihood;
    public double modelLoglikelihood;
    private double[][][] perplexity;
    private HashMap<String, Integer> days;
    private HashMap<Integer, String> dates;
    private HashMap<Integer, String> gridMap;
    private double[][][] nBookingsByGTD;
    public double testPerplexity;

    // utilities
    public Random rand;
    private double[] dets;
    private double[][][] ivnSigma;
    // learning options
    public int nUpdateIterations;

    /**
     * constructor
     *
     * @param n
     * @param r
     */
    public GGMM(int n, Random r) {
        rand = r;
        nUpdateIterations = n;
    }

    public void getLogLikelihood(String model, Booking[] data) {
        modelLoglikelihood = 0;
        for (int i = 0; i < data.length; i++) {
            double pb = 0;
            for (int k = 0; k < K; k++) {
                if (model.equals("ggmm")) {
                    int g = data[i].gridIndex;
                    pb += pi[g][k]
                            * Math.exp(data[i].getGaussianLogLikelihood(mu[k],
                            ivnSigma[k], dets[k]));
                } else if (model.equals("gmm")) {
                    pb += gmmPi[k]
                            * Math.exp(data[i].getGaussianLogLikelihood(mu[k],
                            ivnSigma[k], dets[k]));
                } else {// for the next models
                }
            }
            modelLoglikelihood += Math.log(pb);
        }
    }


    /**
     * learn parameters from data
     *
     * @param nClusters
     * @param data
     */

    public void learnGGMM(int nClusters, Booking[] data) {
        K = nClusters;
        // get dimension
        dim = data[0].features.length;
        // count grid cells;
        HashSet<Integer> grids = new HashSet<Integer>();
        for (int i = 0; i < data.length; i++)
            grids.add(data[i].gridIndex);
        G = grids.size();

        int[][] nElements = new int[G][K];// #elements in each grid cell and
        // each cluster
        int[] nElementsByGrid = new int[G];// #elements in each grid
        int[] nElementsByCluster = new int[G];// #elements in each cluster

        double[][] sums = new double[K][dim];

        // initialization
        System.out.println("initializing by Kmeans");
        int[] cIndexes = kmeans(data);
//        System.out.println("initializing params");
        for (int i = 0; i < data.length; i++) {
            int g = data[i].gridIndex;
            int k = cIndexes[i];
            nElements[g][k]++;
            nElementsByGrid[g]++;
            nElementsByCluster[k]++;
            for (int j = 0; j < dim; j++)
                if (j != Parameter.temporalIndex)
                    sums[k][j] += data[i].features[j];
        }


        pi = new double[G][K];
        mu = new double[K][dim];
        sigma = new double[K][dim][dim];

        double[][] times = new double[K][];

        for (int k = 0; k < K; k++) {
            // pi
            for (int g = 0; g < G; g++) {
                pi[g][k] = (double) nElements[g][k] / nElementsByGrid[g];
            }
            // mu
            for (int j = 0; j < dim; j++)
                if (j != Parameter.temporalIndex)
                    mu[k][j] = sums[k][j] / nElementsByCluster[k];
            times[k] = new double[nElementsByCluster[k]];
            nElementsByCluster[k] = 0;
        }
        // temporal mean
        for (int i = 0; i < data.length; i++) {
            int k = cIndexes[i];
            times[k][nElementsByCluster[k]] = data[i].features[Parameter.temporalIndex];
            nElementsByCluster[k]++;
        }
        for (int k = 0; k < K; k++) {
            mu[k][Parameter.temporalIndex] = Utility.temporalAvg(times[k]);
        }

        // sigma
        for (int i = 0; i < data.length; i++) {
            int k = cIndexes[i];
            double[] diff = new double[dim];
            for (int p = 0; p < dim; p++) {
                if (p != Parameter.temporalIndex)
                    diff[p] = data[i].features[p] - mu[k][p];
                else
                    diff[p] = Utility.temporalMinus(data[i].features[p],
                            mu[k][p]);
            }
            for (int p = 0; p < dim; p++) {
                for (int q = 0; q < dim; q++) {
                    sigma[k][p][q] += (diff[p] * diff[q])
                            / nElementsByCluster[k];
                }
            }
        }

        // determinants and inversions of sigmas
        LUDecomposition luDecomp;
        dets = new double[K];
        ivnSigma = new double[K][][];
        for (int k = 0; k < K; k++) {
            luDecomp = new LUDecomposition(new Array2DRowRealMatrix(sigma[k]));
            dets[k] = luDecomp.getDeterminant();
            ivnSigma[k] = luDecomp.getSolver().getInverse().getData();
        }

        loglikelihood = new double[data.length][K];
        // iteratively updating
        for (int iter = 0; iter < nUpdateIterations; iter++) {
            System.out.printf("iteration %d\n", iter);
            // do hard clustering
            for (int i = 0; i < data.length; i++) {
                int g = data[i].gridIndex;
                // find new cluster
                double maxL = Double.NEGATIVE_INFINITY;
                int maxK = -1;
                for (int k = 0; k < K; k++) {
                    if (nElements[g][k] == 0)
                        loglikelihood[i][k] = Double.NEGATIVE_INFINITY;
                    else
                        loglikelihood[i][k] = Math.log(pi[g][k])
                                + data[i].getGaussianLogLikelihood(mu[k],
                                ivnSigma[k], dets[k]);
                    if (maxL < loglikelihood[i][k]) {
                        maxL = loglikelihood[i][k];
                        maxK = k;
                    }
                }

                // reduce counts and sum of the current cluster
                int currentK = cIndexes[i];// get current cluster
                nElements[g][currentK]--;
                nElementsByCluster[currentK]--;
                for (int p = 0; p < dim; p++) {
                    if (p != Parameter.temporalIndex)
                        sums[currentK][p] -= data[i].features[p];
                }
                // assign new cluster and increase the counts and sum
                cIndexes[i] = maxK;// assign new cluster
                nElements[g][maxK]++;
                nElementsByCluster[maxK]++;
                for (int p = 0; p < dim; p++) {
                    if (p != Parameter.temporalIndex)
                        sums[maxK][p] += data[i].features[p];
                }
            }
            for (int k = 0; k < K; k++) {
                // pi
                for (int g = 0; g < G; g++) {
                    pi[g][k] = (double) nElements[g][k] / nElementsByGrid[g];
                }
                // mu
                if (nElementsByCluster[k] > 0)
                    for (int j = 0; j < dim; j++)
                        if (j != Parameter.temporalIndex)
                            mu[k][j] = sums[k][j] / nElementsByCluster[k];
                times[k] = new double[nElementsByCluster[k]];
                nElementsByCluster[k] = 0;
            }
            // temporal mean
            for (int i = 0; i < data.length; i++) {
                int k = cIndexes[i];
                times[k][nElementsByCluster[k]] = data[i].features[Parameter.temporalIndex];
                nElementsByCluster[k]++;
            }
            for (int k = 0; k < K; k++) {
                mu[k][Parameter.temporalIndex] = Utility.temporalAvg(times[k]);
            }

            // sigma
            for (int k = 0; k < K; k++) {
                if (nElementsByCluster[k] > 0)
                    for (int p = 0; p < dim; p++) {
                        for (int q = 0; q < dim; q++) {
                            sigma[k][p][q] = 0;
                        }
                    }
            }
            for (int i = 0; i < data.length; i++) {
                int k = cIndexes[i];
                if (nElementsByCluster[k] > 0) {
                    double[] diff = new double[dim];
                    for (int p = 0; p < dim; p++) {
                        if (p != Parameter.temporalIndex)
                            diff[p] = data[i].features[p] - mu[k][p];
                        else
                            diff[p] = Utility.temporalMinus(
                                    data[i].features[p], mu[k][p]);
                    }
                    for (int p = 0; p < dim; p++) {
                        for (int q = 0; q < dim; q++) {
                            sigma[k][p][q] += (diff[p] * diff[q])
                                    / nElementsByCluster[k];
                        }
                    }
                }
            }
            // determinants of sigma
            for (int k = 0; k < K; k++) {
                if (nElementsByCluster[k] > 0) {
                    luDecomp = new LUDecomposition(new Array2DRowRealMatrix(
                            sigma[k]));
                    dets[k] = luDecomp.getDeterminant();
                    ivnSigma[k] = luDecomp.getSolver().getInverse().getData();
                }
            }
        }
    }

    public void learnSoftGGMM(int nClusters, Booking[] data) {
        K = nClusters;
        // get dimension
        dim = data[0].features.length;
        // count grid cells;
        HashSet<Integer> grids = new HashSet<Integer>();
        for (int i = 0; i < data.length; i++)
            grids.add(data[i].gridIndex);
        G = grids.size();

        double[][] sumResponsibility = new double[G][K];// #elements in each
        // grid cell and
        // each cluster
        double[] sumResponsibilityByGrid = new double[G];// #elements in each
        // grid
        double[] sumResponsibilityByCluster = new double[K];// #elements in each
        // cluster

        double[][] sums = new double[K][dim];

        // initialization
        int[] cIndexes = kmeans(data);
        for (int i = 0; i < data.length; i++) {
            int g = data[i].gridIndex;
            int k = cIndexes[i];
            sumResponsibility[g][k]++;
            sumResponsibilityByGrid[g]++;
            sumResponsibilityByCluster[k]++;
            for (int j = 0; j < dim; j++)
                if (j != Parameter.temporalIndex)
                    sums[k][j] += data[i].features[j];
        }

        pi = new double[G][K];
        mu = new double[K][dim];
        sigma = new double[K][dim][dim];

        double[][] times = new double[K][];

        for (int k = 0; k < K; k++) {
            // pi
            for (int g = 0; g < G; g++) {
                pi[g][k] = (double) sumResponsibility[g][k]
                        / sumResponsibilityByGrid[g];
            }
            // mu
            for (int j = 0; j < dim; j++)
                if (j != Parameter.temporalIndex)
                    mu[k][j] = sums[k][j] / sumResponsibilityByCluster[k];
            times[k] = new double[(int) sumResponsibilityByCluster[k]];
            sumResponsibilityByCluster[k] = 0;
        }
        // temporal mean
        for (int i = 0; i < data.length; i++) {
            int k = cIndexes[i];
            times[k][(int) sumResponsibilityByCluster[k]] = data[i].features[Parameter.temporalIndex];
            sumResponsibilityByCluster[k]++;
        }
        for (int k = 0; k < K; k++) {
            mu[k][Parameter.temporalIndex] = Utility.temporalAvg(times[k]);
        }

        // sigma
        for (int i = 0; i < data.length; i++) {
            int k = cIndexes[i];
            double[] diff = new double[dim];
            for (int p = 0; p < dim; p++) {
                if (p != Parameter.temporalIndex)
                    diff[p] = data[i].features[p] - mu[k][p];
                else
                    diff[p] = Utility.temporalMinus(data[i].features[p],
                            mu[k][p]);
            }
            for (int p = 0; p < dim; p++) {
                for (int q = 0; q < dim; q++) {
                    sigma[k][p][q] += (diff[p] * diff[q])
                            / sumResponsibilityByCluster[k];
                }
            }
        }
        // determinants and inversions of sigmas
        LUDecomposition luDecomp;
        dets = new double[K];
        ivnSigma = new double[K][][];
        for (int k = 0; k < K; k++) {
            luDecomp = new LUDecomposition(new Array2DRowRealMatrix(sigma[k]));
            dets[k] = luDecomp.getDeterminant();
            ivnSigma[k] = luDecomp.getSolver().getInverse().getData();
        }

        double[][] responsibility = new double[K][data.length];
        double[] allTimes = new double[data.length];
        double min = 24;
        for (int i = 0; i < data.length; i++) {
            allTimes[i] = data[i].features[Parameter.temporalIndex];
            if (allTimes[i] < min)
                min = allTimes[i];
        }
        // iteratively updating
        for (int iter = 0; iter < nUpdateIterations; iter++) {
            // do soft clustering
            for (int g = 0; g < G; g++) {
                sumResponsibilityByGrid[g] = 0;
                for (int k = 0; k < K; k++)
                    sumResponsibility[g][k] = 0;
            }
            for (int k = 0; k < K; k++) {
                sumResponsibilityByCluster[k] = 0;
                for (int p = 0; p < dim; p++) {
                    if (p != Parameter.temporalIndex)
                        sums[k][p] = 0;
                }
            }
            for (int i = 0; i < data.length; i++) {
                int g = data[i].gridIndex;
                // find new cluster
                double norm = 0;
                for (int k = 0; k < K; k++) {
                    responsibility[k][i] = pi[g][k]
                            * Math.exp(data[i].getGaussianLogLikelihood(mu[k],
                            ivnSigma[k], dets[k]));
                    norm += responsibility[k][i];
                }
                for (int k = 0; k < K; k++) {
                    responsibility[k][i] /= norm;
                    sumResponsibility[g][k] += responsibility[k][i];
                    sumResponsibilityByGrid[g] += responsibility[k][i];
                    sumResponsibilityByCluster[k] += responsibility[k][i];
                    for (int p = 0; p < dim; p++) {
                        if (p != Parameter.temporalIndex)
                            sums[k][p] += data[i].features[p]
                                    * responsibility[k][i];
                    }
                }
            }

            // update parameters
            for (int k = 0; k < K; k++) {
                // pi
                for (int g = 0; g < G; g++) {
                    pi[g][k] = (double) sumResponsibility[g][k]
                            / sumResponsibilityByGrid[g];
                }
                // mu
                if (sumResponsibilityByCluster[k] > 0)
                    for (int j = 0; j < dim; j++)
                        if (j != Parameter.temporalIndex)
                            mu[k][j] = sums[k][j]
                                    / sumResponsibilityByCluster[k];
            }
            // temporal mean
            for (int k = 0; k < K; k++) {
                mu[k][Parameter.temporalIndex] = Utility.temporalAvg(allTimes,
                        responsibility[k], min);
            }

            // sigma
            for (int k = 0; k < K; k++) {
                if (sumResponsibilityByCluster[k] > 0)
                    for (int p = 0; p < dim; p++) {
                        for (int q = 0; q < dim; q++) {
                            sigma[k][p][q] = 0;
                        }
                    }
            }
            for (int i = 0; i < data.length; i++) {
                for (int k = 0; k < K; k++) {
                    if (sumResponsibilityByCluster[k] > 0) {
                        double[] diff = new double[dim];
                        for (int p = 0; p < dim; p++) {
                            if (p != Parameter.temporalIndex)
                                diff[p] = data[i].features[p] - mu[k][p];
                            else
                                diff[p] = Utility.temporalMinus(
                                        data[i].features[p], mu[k][p]);
                        }
                        for (int p = 0; p < dim; p++) {
                            for (int q = 0; q < dim; q++) {
                                sigma[k][p][q] += ((diff[p] * diff[q]) / sumResponsibilityByCluster[k])
                                        * responsibility[k][i];
                            }
                        }
                    }
                }
            }
            // determinants of sigma
            for (int k = 0; k < K; k++) {
                if (sumResponsibilityByCluster[k] > 0) {
                    luDecomp = new LUDecomposition(new Array2DRowRealMatrix(
                            sigma[k]));
                    dets[k] = luDecomp.getDeterminant();
                    ivnSigma[k] = luDecomp.getSolver().getInverse().getData();
                }
            }
        }
    }

    /**
     * * clustering by simple kmeans method
     *
     * @param data
     * @return cluster index array
     */
    private int[] kmeans(Booking[] data) {
        try {
            int[] cIndexes = new int[data.length];
            int[] nElements = new int[K];
            double[][] sums = new double[K][dim];
            double[][] means = new double[K][dim];
            // randomly initialize clusters
//            System.out.println("-----Kmeans: initializing");
            for (int k = 0; k < K; k++) {
                int i = rand.nextInt(data.length);
                for (int j = 0; j < dim; j++) {
                    means[k][j] = data[i].features[j];
                }
            }

            for (int i = 0; i < data.length; i++) {
                // assign cluster
                cIndexes[i] = 0;
                double minDistance = data[i].getEucDistance(means[0]);
                for (int k = 1; k < K; k++) {
                    double distance = data[i].getEucDistance(means[k]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        cIndexes[i] = k;
                    }
                }
                // increase count and sum
                nElements[cIndexes[i]]++;
                for (int j = 0; j < dim; j++) {
                    if (j != Parameter.temporalIndex)
                        sums[cIndexes[i]][j] += data[i].features[j];
                }
            }

            // means
            double[][] times = new double[K][];
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < dim; j++) {
                    if (j != Parameter.temporalIndex)
                        means[k][j] = sums[k][j] / nElements[k];
                }
                times[k] = new double[nElements[k]];
                nElements[k] = 0;
            }

            // temporal mean
            for (int i = 0; i < data.length; i++) {
                int k = cIndexes[i];
                times[k][nElements[k]] = data[i].features[Parameter.temporalIndex];
                nElements[k]++;
            }
            for (int k = 0; k < K; k++) {
                means[k][Parameter.temporalIndex] = Utility
                        .temporalAvg(times[k]);
            }

            // update iteratively
            for (int iter = 0; iter < nUpdateIterations; iter++) {
//                System.out.printf("-----Kmeans: iteration %d/%d\n", iter,
//                        nUpdateIterations);
                for (int i = 0; i < data.length; i++) {
                    // get current cluster index
                    int currentK = cIndexes[i];
                    // reduce count and sum of currentCluster
                    nElements[currentK]--;
                    for (int j = 0; j < dim; j++) {
                        if (j != Parameter.temporalIndex)
                            sums[currentK][j] -= data[i].features[j];
                    }
                    // find new cluster
                    int newK = -1;
                    double minDistance = Double.MAX_VALUE;
                    for (int k = 0; k < K; k++) {
                        double distance = data[i].getEucDistance(means[k]);
                        if (distance < minDistance) {
                            minDistance = distance;
                            newK = k;
                        }
                    }
                    // reassign cluster index
                    cIndexes[i] = newK;
                    // increase count and sum of newCluster
                    nElements[newK]++;
                    for (int j = 0; j < dim; j++) {
                        if (j != Parameter.temporalIndex)
                            sums[newK][j] += data[i].features[j];
                    }
                }
                // update means
                for (int k = 0; k < K; k++) {
                    for (int j = 0; j < dim; j++) {
                        if (j != Parameter.temporalIndex)
                            means[k][j] = sums[k][j] / nElements[k];
                    }
                    times[k] = new double[nElements[k]];
                    nElements[k] = 0;
                }
                // update temporal mean
                for (int i = 0; i < data.length; i++) {
                    int k = cIndexes[i];
                    times[k][nElements[k]] = data[i].features[Parameter.temporalIndex];
                    nElements[k]++;
                }
                for (int k = 0; k < K; k++) {
                    means[k][Parameter.temporalIndex] = Utility
                            .temporalAvg(times[k]);
                }
            }
            return cIndexes;
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return null;
    }

    public void outputParameters(String outputPath, String model) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(outputPath
                    + "/parameters_" + model + ".txt"));
//            if (model.equals("ggmm")) {
                bw.write("#grid cells = " + G + "\n");

                for (int g = 0; g < G; g++) {
                    bw.write("pi_" + g + " = " + pi[g][0]);
                    for (int k = 1; k < K; k++) {
                        bw.write(" " + pi[g][k]);
                    }
                    bw.write("\n");
                }

            bw.write("#clusters = " + K + "\n");

            for (int k = 0; k < K; k++) {
                bw.write("mu_" + k + " = " + mu[k][0]);
                for (int j = 1; j < dim; j++)
                    bw.write(" " + mu[k][j]);
                bw.write("\n");
                bw.write("sigma_" + k + " =");
                for (int p = 0; p < dim; p++) {
                    bw.write(" " + sigma[k][p][p]);
                }
                bw.write("\n");
            }
            bw.write("\n");
            bw.close();



        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }
}

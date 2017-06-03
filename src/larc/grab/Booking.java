public class Booking {
	public String day;
	public String gridName;
	public int gridIndex;
	public double[] features;


	public double getEucDistance(double[] p) {
		double d = 0;
		for (int i = 0; i < features.length; i++) {
			if (i != Parameter.temporalIndex)
				d += Math.pow(this.features[i] - p[i], 2);
			else
				d += Math.pow(Utility.temporalMinus(this.features[i], p[i]), 2);
		}
		return Math.sqrt(d);
	}

	public double getGaussianLogLikelihood(double[] m, double[][] ivnsigma, double detSigma) {
		double[] diff = new double[features.length];
		for (int i = 0; i < features.length; i++) {
			if (i != Parameter.temporalIndex)
				diff[i] = this.features[i] - m[i];
			else
				diff[i] = Utility.temporalMinus(this.features[i], m[i]);
		}
		double loglikelihood = 0;
		for (int i = 0; i < features.length; i++) {
			for (int j = 0; j < features.length; j++) {
				loglikelihood += diff[i] * diff[j] * ivnsigma[i][j];
			}
		}
		loglikelihood /= (-2);		
		loglikelihood -= (features.length / 2.0) * Math.log(2 * Math.PI);
		loglikelihood -= 0.5 * Math.log(detSigma);
		return loglikelihood;
	}
}
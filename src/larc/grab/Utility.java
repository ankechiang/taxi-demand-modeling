public class Utility {
	/**
	 * return x - y
	 *
	 * @param x
	 * @param y
	 * @return
	 */
	static double temporalMinus(double x, double y) {
		if (x < 12) {
			if (y <= x) {
				return x - y;
			} else if (x < y && y < x + 12) {
				return x - y;
			} else {
				return x + 24 - y;
			}
		} else {
			if (y < x - 12) {
				return x - 24 - y;
			} else if (x - 12 <= y && y < x) {
				return x - y;
			} else {
				return x - y;
			}
		}
	}

	/**
	 * return weighted avg of a and b
	 *
	 * @param a
	 * @param b
	 * @param aWeight
	 * @param bWeight
	 * @return
	 */
	static double temporalAvg(double a, double aWeight, double b, double bWeight) {
		if (aWeight + bWeight <= Double.MIN_VALUE) {
			System.out.println("Singular case found!");
			System.exit(-1);
		}

		if (aWeight <= Double.MIN_VALUE)
			return b / bWeight;
		if (bWeight <= Double.MIN_VALUE)
			return a / aWeight;
		double x, y, xWeight, yWeight;
		if (a < b) {
			x = a;
			xWeight = aWeight;
			y = b;
			yWeight = bWeight;
		} else {
			x = b;
			xWeight = bWeight;
			y = a;
			yWeight = aWeight;
		}

		if (y - x < 12)
			return (x * xWeight + y * yWeight) / (xWeight + yWeight);
		else {
			double m = (xWeight * x + yWeight * y + yWeight * 24)
					/ (xWeight + yWeight);
			if (x <= m && m < 24)
				return m;
			m = (xWeight * x - xWeight * 24 + yWeight * y)
					/ (xWeight + yWeight);
			return m;
		}
	}

	/**
	 * return avg of an array of temporal time points
	 *
	 * @param times
	 * @return
	 */
	static double temporalAvg(double[] times) {
		double min = Double.POSITIVE_INFINITY;
		for (int i = 0; i < times.length; i++) {
			if (times[i] < min)
				min = times[i];
		}
		double a = 0;
		int n = 0;
		double b = 0;
		int m = 0;
		for (int i = 0; i < times.length; i++) {
			if (times[i] < min + 12) {
				a += times[i];
				n++;
			} else {
				b += times[i];
				m++;
			}
		}
		if (m == 0)
			return a / n;
		return temporalAvg(a / n, n, b / m, m);
	}

	/**
	 * return avg of an array of temporal time points
	 *
	 * @param times
	 * @return
	 */
	static double temporalAvg(double[] times, double[] weight) {
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < times.length; i++) {
			if (times[i] < min)
				min = times[i];
			if (max < times[i])
				max = times[i];
		}
		double a = 0;
		double n = 0;
		double b = 0;
		double m = 0;
		for (int i = 0; i < times.length; i++) {
			if (times[i] < min + 12) {
				a += times[i] * weight[i];
				n += weight[i];
			} else {
				b += times[i] * weight[i];
				m += weight[i];
			}
		}
		if (max <= min + 12)
			return a / n;
		return temporalAvg(a / n, n, b / m, m);
	}

	static double temporalAvg(double[] times, double[] weight, double min) {
		// System.out.printf("in temporalAvg: min = %f max = %f\n", min, max);
		double a = 0;
		double n = 0;
		double b = 0;
		double m = 0;
		for (int i = 0; i < times.length; i++) {
			if (times[i] < min + 12) {
				a += times[i] * weight[i];
				n += weight[i];
			} else {
				b += times[i] * weight[i];
				m += weight[i];
			}
		}
		/*
		 * System.out.printf("in temporalAvg: a = %f b = %f n = %f m = %f\n", a,
		 * b, n, m);
		 */
		if (n <= Double.MIN_VALUE)
			return b / m;
		if (m <= Double.MIN_VALUE)
			return a / n;
		return temporalAvg(a / n, n, b / m, m);
	}

}

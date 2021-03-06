import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import java.io.*;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import static org.apache.spark.sql.functions.*;

public class SparkAppMain {
    public static <T> void printNestedDoubleList(List<List<T>> list) {
        for(List<T> l : list) {
            System.out.println(l);
        }
    }
    public static List<Tuple2<LocalDate, Double>> readHistory(File file) throws FileNotFoundException, IOException {
       // System.out.println("-----------------------------------");
     //   System.out.println(file.getName());
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("d-MMM-yy");
        BufferedReader br = new BufferedReader(new FileReader(file));
        List<String> lines = new ArrayList<String>();
        String st;
        int i=0;
        while((st = br.readLine())!=null) {
           // System.out.println(st);
            ++i;
            if(i==1) continue;
            lines.add(st);
        }
       // System.out.println("-----------------------------------");
        List<Tuple2<LocalDate, Double>> ret = new ArrayList<Tuple2<LocalDate, Double>>();
        for(String line : lines) {
            String[] cols = line.split(",");
           // System.out.println(cols[0]);
            LocalDate date =  LocalDate.parse(cols[0], formatter);
            double value = Double.parseDouble(cols[4]);
            ret.add(new Tuple2<>(date, value));
        }
        Collections.reverse(ret);
        return ret;
    }
    public static List<Tuple2<LocalDate, Double>> trimToRegion(List<Tuple2<LocalDate, Double>> history, LocalDate start, LocalDate end) {
        history.removeIf(elem -> elem._1.isBefore(start));
        history.removeIf(elem -> !((elem._1.isBefore(end)) || (elem._1.isEqual(end))));
        if(history.get(0)._1!=start) {
            history.add(0, new Tuple2<>(start, history.get(0)._2));
        }
        if(history.get(history.size()-1)._1!=end) {
            history.add(new Tuple2<>(end, history.get(history.size()-1)._2));
        }
        return history;
    }
    public static List<Tuple2<LocalDate, Double>> fillInHistory(List<Tuple2<LocalDate, Double>> history, LocalDate start, LocalDate end) {
        List<Tuple2<LocalDate, Double>> cur = new ArrayList<>(history);
        List<Tuple2<LocalDate, Double>> filled = new  ArrayList<Tuple2<LocalDate, Double>>();
        LocalDate curDate = start;
        while (curDate.isBefore(end)) {
            List<Tuple2<LocalDate, Double>> subList = cur.subList(1, cur.size()-1);
            if(!(subList.isEmpty()) && subList.get(0)._1.equals(curDate)) {
                cur.remove(0);
            }
            filled.add(new Tuple2<>(curDate, cur.get(0)._2));
            curDate = curDate.plusDays(1);
            if(curDate.getDayOfWeek().getValue() > 5)
                curDate = curDate.plusDays(2);
        }
        return filled;
    }
    public static List<Double> twoWeekReturns(List<Tuple2<LocalDate, Double>> history) {
       // System.out.println(history.size());
        List<Double> ret = new ArrayList<Double>();
        int i=0;
        while(i<= history.size()-1 && (i+10)<=history.size()) {
            List<Tuple2<LocalDate, Double>> subList = history.subList(i, i+10);
            double next = subList.get(subList.size()-1)._2;
            double prev = subList.get(0)._2;
            double elem = (next-prev);
            ret.add(elem);
            i++;
        }
//        System.out.println(ret);
        return ret;
    }
    public static Tuple2<List<List<Double>>, List<List<Double>>> readStocksAndFactors() throws IOException {
        LocalDate start = LocalDate.of(2009, 10, 23);
        LocalDate end = LocalDate.of(2014, 10, 23);

        File stocksDir = new File("data/stocks/");
        List<File> files =  Arrays.asList(stocksDir.listFiles());
        List<List<Tuple2<LocalDate, Double>>> allStocks = new ArrayList<List<Tuple2<LocalDate, Double>>>();
        for(File f : files) {
            allStocks.add(readHistory(f));
        }
        allStocks.removeIf(ll -> ll.size() < 260 * 5 + 10);
        String factorsPrefix = "data/factors/";
        String factorNames[] =  {"NASDAQ%3ATLT.csv", "NYSEARCA%3ACRED.csv", "NYSEARCA%3AGLD.csv"};
        List<File> factorFiles = new ArrayList<File>();
        for(String s : factorNames) {
            factorFiles.add(new File(factorsPrefix + s));
        }
        List<List<Tuple2<LocalDate, Double>>> rawFactors = new ArrayList<List<Tuple2<LocalDate, Double>>>();
        for(File f : factorFiles) {
            rawFactors.add(readHistory(f));
        }

        List<List<Tuple2<LocalDate, Double>>> stocks = allStocks.stream().map(s -> trimToRegion(s, start, end)).map(s -> fillInHistory(s, start, end)).collect(Collectors.toList());
        List<List<Tuple2<LocalDate, Double>>> factors = rawFactors.stream().map(f -> trimToRegion(f, start, end)).map(f -> fillInHistory(f, start, end)).collect(Collectors.toList());
        List<List<Double>> stocksReturns = stocks.stream().map(SparkAppMain::twoWeekReturns).collect(Collectors.toList());
        List<List<Double>> factorsReturns = factors.stream().map(SparkAppMain::twoWeekReturns).collect(Collectors.toList());
        return new Tuple2<List<List<Double>>, List<List<Double>>>(stocksReturns, factorsReturns);
    }
    public static List<List<Double>> factorMatrix(List<List<Double>> histories) {
        List<List<Double>> mat = new ArrayList<List<Double>>();
        List<Double> head = histories.get(0);
        for(int i=0; i< head.size(); ++i) {
            int finalI = i;
            mat.add(i, histories.stream().map(e -> e.get(finalI)).collect(Collectors.toList()));
        }
        return mat;
    }
    public static List<Double> featurize(List<Double> factorReturns) {
       List<Double> squaredReturns = factorReturns.stream().map(x -> Math.signum(x)*x*x).collect(Collectors.toList());
       List<Double> squaredRootedReturns = factorReturns.stream().map(x -> Math.signum(x) * Math.sqrt(Math.abs(x))).collect(Collectors.toList());
       return Stream.of(factorReturns, squaredReturns, squaredRootedReturns).flatMap(Collection::stream).collect(Collectors.toList());
    }
    public static double[][] doubleNestedListto2DArray(List<List<Double>> list) {
        double[][] array = new double[list.size()][];
        for(int i=0; i<array.length; ++i) {
            array[i] = new double[list.get(i).size()];
        }
        for (int i = 0; i < array.length; i++) {
            for(int j=0; j<list.get(i).size(); ++j) {
                array[i][j] = list.get(i).get(j);
            }
        }
        return array;
    }
    public static OLSMultipleLinearRegression linearModel(List<Double> instrument, List<List<Double>> factorMatrix) {
        OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
        double[] instrumentArr = instrument.stream().mapToDouble(Double::doubleValue).toArray();
        regression.newSampleData(instrumentArr, doubleNestedListto2DArray(factorMatrix));
        return regression;
    }
    public static List<List<Double>> computeFactorWeights(List<List<Double>> stockReturns, List<List<Double>> factorFeatures) {
        return stockReturns.stream().map(s -> linearModel(s, factorFeatures)).map(s -> DoubleStream.of(s.estimateRegressionParameters()).boxed().collect(Collectors.toList())).collect(Collectors.toList());
    }
    public static List<Double> trialReturns(Long seed, int numTrials, List<List<Double>> instruments, List<Double> factorMeans, double[][] factorCovariances, SparkSession spark) {
       // System.out.println("------------------------TRIAL RETURNS-----------------------------");
        MersenneTwister rand = new MersenneTwister(seed);
        MultivariateNormalDistribution multivariateNormalDistribution = new MultivariateNormalDistribution(rand, factorMeans.stream().mapToDouble(d -> d).toArray(), factorCovariances);
        List<Double> tReturns = new ArrayList<Double>();
        for(int i=0; i<numTrials; ++i) {
           double[] trialFactorReturns = multivariateNormalDistribution.sample();
           //System.out.println(Arrays.toString(trialFactorReturns));
           List<Double> trialFeatures = featurize(DoubleStream.of(trialFactorReturns).boxed().collect(Collectors.toList()));
          // System.out.println(trialFeatures);
           tReturns.add(i, trialReturn(trialFeatures, instruments));
        }
       // System.out.println(tReturns);
        return tReturns;
    }
    public static double trialReturn(List<Double> trial, List<List<Double>> instruments) {
       // printNestedDoubleList(instruments);
        //System.out.println("------------------------TRIAL RETURN-----------------------------");
        double totalReturn = 0.0; int size = 0;
        for(List<Double> instrument : instruments) {
            totalReturn+= instrumentTrialReturn(instrument, trial);
        }
//        System.out.println(totalReturn);
        //if(instruments.size()==0) System.out.println("----------------ZERO----------------------");

        return totalReturn/instruments.size();
    }
    public static double instrumentTrialReturn(List<Double> instrument, List<Double> trial) {
//       System.out.println("------------------------INSTRUMENT TRIAL RETURN-----------------------------");
        double itr = 0.0;
       if(!Double.isNaN(instrument.get(0))) itr = instrument.get(0);
        int i=0;
        while (i<trial.size()) {
            double t_el = trial.get(i), i_el = instrument.get(i+1);
            if(!Double.isNaN(t_el) && !Double.isNaN(i_el)) {
                //System.out.println(itr);
                itr += trial.get(i) * instrument.get(i+1);
//                System.out.println(itr);
            }
            ++i;
        }
        //System.out.println(itr);
        return itr;
    }
    public static Dataset<Double> computeTrialReturns(List<List<Double>> stocksReturns, List<List<Double>> factorsReturns, Long baseSeed, int numTrials, int parallelism, SparkSession spark) {
        List<List<Double>> factorMat = factorMatrix(factorsReturns);
//        System.out.println("----------------FACTOR MATRIX---------------------");
//        printNestedDoubleList(factorMat);
       double[][] factorCov =  new Covariance(doubleNestedListto2DArray(factorMat)).getCovarianceMatrix().getData();
//        System.out.println("----------------FACTOR COVARIANCE---------------------");
//       for(double[] row : factorCov) {
//           System.out.println(Arrays.toString(row));
//       }
       List<Double> factorMeans = factorsReturns.stream().map(factor -> {
           double sum = 0.0;
           for(double f : factor) {
               sum +=f;
           }
           return sum/factor.size();
       }).collect(Collectors.toList());
      //  System.out.println("----------------FACTOR MEANS---------------------");
       // System.out.println(factorMeans);
       List<List<Double>> factorFeatures = factorMat.stream().map(SparkAppMain::featurize).collect(Collectors.toList());
        System.out.println("----------------FACTOR WEIGHTS--------------------");
       // printNestedDoubleList(factorFeatures);
      List<List<Double>> factorWeights = computeFactorWeights(stocksReturns, factorFeatures);
      List<Long> seeds = new ArrayList<Long>();
      for(long i=baseSeed; i<=baseSeed + parallelism; ++i) {
          seeds.add(i);
      }
      Dataset<Long> seedsDS = spark.createDataset(seeds, Encoders.LONG()).repartition(parallelism);
    //  seedsDS.show();
//        for(long seed: seeds){
//            trialReturns(seed, numTrials/parallelism, factorWeights, factorMeans, factorCov, spark);
//        }
      return seedsDS.flatMap((Long seed) -> trialReturns(seed, numTrials/parallelism, factorWeights, factorMeans, factorCov, spark).iterator(), Encoders.DOUBLE());
    }
    public static double fivePercentVaR(Dataset<Double> trials) {
        double[] quantiles = trials.stat().approxQuantile("value", new double[] {0.05}, 0.0);
        return quantiles.length!=0 ? quantiles[quantiles.length-1] : 0.0;
    }
    public static double fivePercentCVar(Dataset<Double> trials) {
        Dataset<Double> topLosses = trials.orderBy("value").limit(Math.max((int)(trials.count())/20, 1));
        double sum =  (double)topLosses.agg(sum("value")).first().get(0);
        return sum/topLosses.count();
    }
    public  static Tuple2<Double, Double> bootstrappedConfidenceInterval(Dataset<Double> trials, String riskType, int numResamples, double probability) {
        List<Integer> range = new ArrayList<>();
        for(int i=0; i<numResamples; ++i) {
            range.add(i);
        }
       List<Double> stats =  range.stream().map(i -> {
            Dataset<Double> resample = trials.sample(true, 1.0);
            if(riskType.equals("var"))
                return fivePercentVaR(resample);
            else
                return fivePercentCVar(resample);
        }).collect(Collectors.toList());
        Collections.sort(stats);
        int lowerIndex = (int)(numResamples * probability / 2 -1);
        int upperIndex = (int)(Math.ceil(numResamples * (1-probability)/2));
        return new Tuple2<>(stats.get(lowerIndex), stats.get(upperIndex));
    }
    public static int countFailures(List<List<Double>> stocksReturns, double valueAtRisk) {
        int failures = 0;
        List<Double> head = stocksReturns.get(0);
        for(int i=0; i<head.size(); ++i) {
            int finalI = i;
            double loss = stocksReturns.stream().map(e -> e.get(finalI)).mapToDouble(j -> j).sum();
            if(loss < valueAtRisk)
                failures +=1;
        }
        return  failures;
    }

    public static double kupiecTestStatistic(int total, int failures, double confidenceLevel) {
        double failureRatio = ((double) failures)/total;
        double logNumer = (total - failures) * Math.log1p(-confidenceLevel) + failures * Math.log(confidenceLevel);
        double logDenom = (total - failures) * Math.log1p(-failureRatio) + failures * Math.log(failureRatio);
        return -2*(logNumer - logDenom);
    }

    public static double kupiecTestPValue(List<List<Double>> stocksReturns, double valueAtRisk, double confidenceLevel) {
        int failures = countFailures(stocksReturns, valueAtRisk);
        int total = stocksReturns.get(0).size();
        double testStatistic = kupiecTestStatistic(total, failures, confidenceLevel);
        return 1 - new ChiSquaredDistribution(1.0).cumulativeProbability(testStatistic);
//        return testStatistic;
    }


    public static void main(String[] args) throws IOException {
        SparkSession sparkSession = SparkSession.builder().getOrCreate();
        Tuple2<List<List<Double>>, List<List<Double>>> stocksFactorsReturns = readStocksAndFactors();
        int numTrials = 100000;
        int parallelism = 100;
        long baseSeed = 1001L;
        Dataset<Double> trials = computeTrialReturns(stocksFactorsReturns._1, stocksFactorsReturns._2, baseSeed, numTrials, parallelism, sparkSession);
//       trials = trials.filter(not(isnan(trials.col("value"))));
//        trials.show();
        trials.cache();
        double valueAtRisk = fivePercentVaR(trials);
        double conditionalValueAtRisk = fivePercentCVar(trials);
        System.out.println("VaR 5%: " + valueAtRisk);
        System.out.println("CVaR 5%: " + conditionalValueAtRisk);
//        Tuple2<Double, Double> varConfidenceInterval = bootstrappedConfidenceInterval(trials, "var", 100, 0.05);
//        Tuple2<Double, Double> cvarConfidenceInterval = bootstrappedConfidenceInterval(trials, "cvar", 100, 0.05);
//        System.out.println("VaR confidence interval: " + varConfidenceInterval);
//        System.out.println("CVaR confidence interval: " + cvarConfidenceInterval);
        System.out.println("Kupiec test p-value: " + kupiecTestPValue(stocksFactorsReturns._1, valueAtRisk, 0.05));
    }
}

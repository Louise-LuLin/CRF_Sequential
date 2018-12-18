package gLearner;

//import edu.umass.cs.mallet.grmm.types.*;
import cc.mallet.grmm.types.FactorGraph;
import gLearner.GraphLearner;
import gLearner.SeqAnalyzer;
import gLearner.String4Learning;

import java.util.*;
import java.util.concurrent.SynchronousQueue;

/**
 *  This is a class putting node and edge features into table factors.
 */

public class CRF {

    //Sequence analyzer to access all data and vocabulary
    SeqAnalyzer m_seq;

    public CRF( SeqAnalyzer seq){
        m_seq = seq;
    }

    public SeqAnalyzer getSeq(){ return this.m_seq; }

    private double[] calcAcc(ArrayList<int[]> true_labels, ArrayList<ArrayList<Integer>> pred_labels ){
        double[] accs = new double[3]; //all_acc, phrase_acc, out_acc
        if(true_labels.size() != pred_labels.size()) {
            System.err.format("[Err]Prediction has different length than ground truth.\n");
            return accs;
        }

        double[] counts = new double[6]; //all_correct, phrase_correct, out_correct, all_count, phrase_count, out_count
        for(int i = 0; i < true_labels.size(); i++){
            boolean correctFlag = false;
            int label_idx_true, label_idx_pred;
            String label_true;
            String label_pred;
            for(int j = 0; j < true_labels.get(i).length; j++){
                label_idx_true = true_labels.get(i)[j];
                label_true = m_seq.getLabelNames().get(label_idx_true);
                label_idx_pred = pred_labels.get(i).get(j);
                label_pred = m_seq.getLabelNames().get(label_idx_pred);
                //If the character is a beginning-of-phrase.
                if(label_true.startsWith("b")){
                    counts[3+1] += 1;
                    if(label_idx_true == label_idx_pred){
                        if(correctFlag)
                            counts[1] += 1;
                        correctFlag = true;
                    } else {
                        if(correctFlag){
                            if(!label_true.substring(2).equals(
                                    m_seq.getLabelNames().get(pred_labels.get(i).get(j-1)).substring(2))) //special case
                                counts[1] += 1;
                        }
                        correctFlag = false;
                    }
                }//If the character is an inside-of-phrase.
                else if(label_true.startsWith("i")){
                    if(label_idx_pred != label_idx_true)
                        correctFlag = false;
                }//If the character is an out-of-phrase.
                else if(label_true.startsWith("o")){
                    counts[3+2] += 1;
                    if(label_idx_pred == label_idx_true){
                        counts[2] += 1;
                        if(correctFlag){
                            counts[1] += 1;
                            correctFlag = false;
                        }

                    } else {
                        if(correctFlag){
                            if(!label_pred.substring(2).equals(
                                    m_seq.getLabelNames().get(pred_labels.get(i).get(j-1)).substring(2))) //special case
                                counts[1] += 1;
                            correctFlag = false;
                        }

                    }

                }
            }
            //For the case where the phrase is at the end of a string.
            if(correctFlag)
                counts[1] += 1;
        }
        counts[0] = counts[1] + counts[2];
        counts[3] = counts[3+1] + counts[3+2];

        for(int i = 0; i < 3; i++)
            accs[i] = counts[i]/counts[i+3];

        return accs;
    }

    public void activeLearning(String prefix, int maxIter, int train_k, int test_k, int query_k, int tuple_k){
        //get train and test index
        m_seq.genTrainTestIdx(prefix, train_k, test_k);
        ArrayList<Integer> train_idx = m_seq.loadIdx(prefix, "train", train_k);
        ArrayList<Integer> test_idx = m_seq.loadIdx(prefix, "test", test_k);
        ArrayList<Integer> candidate_idx = m_seq.loadIdx(prefix, "candidate",
                m_seq.getStrings().size() - train_k - test_k);
        System.out.format("[Info]Initial: train_size=%d, test_size=%d, candidate_size=%d\n",
                train_idx.size(), test_idx.size(), candidate_idx.size());

        //initial train and test
        Map<Integer, Double> weights = new TreeMap<>();

        ArrayList<int[]> train_label = new ArrayList<>();
        ArrayList<String4Learning> training_data = new ArrayList<>();
        for(int i = 0; i < train_idx.size(); i++){
            train_label.add(m_seq.getLabels().get(train_idx.get(i)));
            training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(train_idx.get(i)), "train", weights));
        }

        ArrayList<int[]> test_label = new ArrayList<>();
        ArrayList<Sequence> testing_seq = new ArrayList<>();
        for(int i = 0; i < test_idx.size(); i++){
            test_label.add(m_seq.getLabels().get(test_idx.get(i)));
            testing_seq.add(m_seq.getSequences().get(test_idx.get(i)));
        }

        //active learning
        double[] acc;
        GraphLearner m_graphLearner = null;
        for(int i = 0 ; i < query_k; i++){

            if(i%10 == 0) {

                System.out.format("==========\n[Info]Active query %d samples: train size = %d, test size = %d...\n",
                        i, training_data.size(), testing_seq.size());

                // Build up a graph learner and train it using training data.
                m_graphLearner = new GraphLearner(training_data);

                // Train
                long start = System.currentTimeMillis();
                ArrayList<ArrayList<Integer>> trainPrediction = m_graphLearner.doTraining(maxIter);
                double acc_cur = calcAcc(train_label, trainPrediction)[0];
                System.out.format("[Info]cur train acc: %f\n", acc_cur);

                weights = m_graphLearner.getWeights();

                // Apply the trained model to the test set.
                ArrayList<ArrayList<Integer>> testPrediction = new ArrayList<>();
                ArrayList<Integer> pred_tmp;
                FactorGraph testGraph;
                int j = 0;
                for (int l = 0; l < testing_seq.size(); l++) {
                    Sequence seq = testing_seq.get(l);
                    testGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));
                    pred_tmp = m_graphLearner.doTesting(testGraph);

                    testPrediction.add(pred_tmp);
                }
                acc = calcAcc(test_label, testPrediction);

                System.out.format("[Stat]Train/test finished in %.2f seconds: acc_all = %.2f, acc_phrase = %.2f, acc_out = %.2f\n",
                        (System.currentTimeMillis() - start) / 1000.0, acc[0], acc[1], acc[2]);
            }


            if(tuple_k == 0){//choose a random one
                Random r = new Random();
                int random_j = r.nextInt(candidate_idx.size());
                train_label.add(m_seq.getLabels().get(candidate_idx.get(random_j)));
                training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(candidate_idx.get(random_j)), "train", weights));
                candidate_idx.remove(random_j);
            } else if(tuple_k >= 50){//choose the one with minimum confidence
                System.out.format("-- query %d\n", i);
                double min = Double.MAX_VALUE;
                int uncertain_j = 0;
                FactorGraph tmpGraph;
                double tmpConficence;
                for(int j = 0; j < candidate_idx.size(); j++){
                    Sequence seq = m_seq.getSequences().get(candidate_idx.get(j));
                    tmpGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));
                    tmpConficence = m_graphLearner.calcConfidence(tmpGraph);
                    if(tmpConficence < min){
                        min = tmpConficence;
                        uncertain_j = j;
                    }
                }

                train_label.add(m_seq.getLabels().get(candidate_idx.get(uncertain_j)));
                training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(candidate_idx.get(uncertain_j)), "train", weights));
                candidate_idx.remove(uncertain_j);
            } else {//choose sub sequence
                System.out.format("-- query %d\n", i);
                double min = Double.MAX_VALUE;
                int uncertain_j = 0, uncertain_k=0;
                FactorGraph tmpGraph;
                double[] tuple_confidence;
                for(int j = 0; j < candidate_idx.size(); j++){
                    Sequence seq = m_seq.getSequences().get(candidate_idx.get(j));
                    tmpGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));
                    tuple_confidence = m_graphLearner.calcTupleConfidence(tmpGraph, 1);
                    for(int k = 0; k < tuple_confidence.length; k++){
                        if(tuple_confidence[k] < min){
                            min = tuple_confidence[k];
                            uncertain_j = j;
                            uncertain_k = k;
                        }
                    }
                }

                train_label.add(Arrays.copyOfRange(m_seq.getLabels().get(candidate_idx.get(uncertain_j)), uncertain_k, tuple_k));
                training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(candidate_idx.get(uncertain_j)).getSubSeq(uncertain_k, tuple_k),
                        "train", weights));
                candidate_idx.remove(uncertain_j);
            }

        }

    }

    public void crossValidation(int k, String prefix, int maxIter){
        double[][] acc = new double[k][3];

        int[] masks = new int[m_seq.getStrings().size()];
        Random rand = new Random();
        for(int i = 0; i < masks.length; i++)
            masks[i] = rand.nextInt(k);

        System.out.format("[Info]Start RANDOM cross validation...\n");        
        
        //use this loop to iterate all the folders, set train and test
        for(int i = 0; i < k; i++){
        	ArrayList<int[]> train_label = new ArrayList<>();
            ArrayList<String4Learning> training_data = new ArrayList<>();

        	ArrayList<int[]> test_label = new ArrayList<>();
            ArrayList<Sequence> testing_seq = new ArrayList<>();

            Map<Integer, Double> weights = new TreeMap<>();

            for(int j = 0; j < masks.length; j++) {
                if (masks[j] == i) {
                    test_label.add(m_seq.getLabels().get(j));
                    testing_seq.add(m_seq.getSequences().get(j));
                } else {
                    train_label.add(m_seq.getLabels().get(j));
                    training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(j), "train", weights));
                }
            }

            System.out.format("==========\n[Info]Fold No. %d: train size = %d, test size = %d...\n",
                    i, training_data.size(), testing_seq.size());
            
            // Build up a graph learner and train it using training data.
            GraphLearner m_graphLearner = new GraphLearner(training_data);

            // Train
            long start = System.currentTimeMillis();
            ArrayList<ArrayList<Integer>> trainPrediction = m_graphLearner.doTraining(maxIter);
            double acc_cur = calcAcc(train_label, trainPrediction)[0];
            System.out.format("cur train acc: %f\n", acc_cur);

//            m_graphLearner.SaveWeights(String.format("%s/weights.txt", prefix));
            weights = m_graphLearner.getWeights();

            // Apply the trained model to the test set.
            ArrayList<ArrayList<Integer>> testPrediction = new ArrayList<>();
            ArrayList<ArrayList<Integer>> testTmp = new ArrayList<>();
            ArrayList<int[]> testTrue = new ArrayList<>();
            ArrayList<Integer> pred_tmp;
            FactorGraph testGraph;
            int j=0;
            for(int l = 0; l < testing_seq.size(); l++) {
                Sequence seq = testing_seq.get(l);
                testGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));
                pred_tmp = m_graphLearner.doTesting(testGraph);

                testPrediction.add(pred_tmp);

                testTmp.clear();
                testTrue.clear();
                testTmp.add(pred_tmp);
                testTrue.add(test_label.get(l));
                double[] acc_tmp = calcAcc(testTrue, testTmp);
                if(acc_tmp[0] < 0.8) {
                    System.out.format("===== bad =====\n");
                    System.out.format("Token: %s\n", seq.getContent());
                }
                if(acc_tmp[0] > 0.95) {
                    System.out.format("===== good =====\n");
                    System.out.format("Token: %s\n", seq.getContent());
                }
                if(acc_tmp[0] < 0.8 || acc_tmp[0] > 0.95) {
                    System.out.format("True: ");
                    int[] labels = seq.getLabels();
                    for(int a = 0; a < labels.length; a++)
                        System.out.format("%s,", m_seq.getLabelNames().get(labels[a]));
                    System.out.format("\n");

                    System.out.format("Pred: ");
                    for(int a = 0; a < pred_tmp.size(); a++)
                        System.out.format("%s,", m_seq.getLabelNames().get(pred_tmp.get(a)));
                    System.out.format("\n");
                }
//                System.out.format("[debug]-- test sample %d predicted label: %d, %d, %d, %d, %d\n", j++,
//                        pred_tmp.get(0), pred_tmp.get(1), pred_tmp.get(2), pred_tmp.get(3), pred_tmp.get(4));
            }
            acc[i] = calcAcc(test_label, testPrediction);

            System.out.format("[Stat]Train/test finished in %.2f seconds: acc_all = %.2f, acc_phrase = %.2f, acc_out = %.2f\n",
                    (System.currentTimeMillis()-start)/1000.0, acc[i][0], acc[i][1], acc[i][2]);
        }

        double[] mean = new double[3];
        double[] var = new double[3];
        for(int i = 0; i < k; i++){
            for(int j = 0; j < 3; j++)
                mean[j] += acc[i][j];
        }
        for(int j = 0; j < 3; j++)
            mean[j] /= k;

        for(int i = 0; i < k; i++){
            for(int j = 0; j < 3; j++){
                var[j] += (acc[i][j] - mean[j]) * (acc[i][j] - mean[j]);
            }
        }
        for(int j = 0; j < 3; j++)
            var[j] = Math.sqrt(var[j]/k);

        System.out.format("[Stat]Accuracy:\n-- all: %.3f+/-%.3f\n-- phrase: %.3f+/-%.3f\n-- out: %.3f+/-%.3f\n",
                mean[0], var[0], mean[1], var[1], mean[2], var[2]);

    }

}

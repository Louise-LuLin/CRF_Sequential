import edu.umass.cs.mallet.grmm.types.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 *  This is a class putting node and edge features into table factors.
 */

public class CRF {

    //Sequence analyzer to access all data and vocabulary
    SeqAnalyzer m_seq;

    //train and test
    ArrayList<String> m_train_string;
    ArrayList<String> m_test_string;
    ArrayList<ArrayList<Integer>> m_train_label;
    ArrayList<ArrayList<Integer>> m_test_label;

    public CRF( SeqAnalyzer seq){
        m_seq = seq;
    }

    public SeqAnalyzer getSeq(){ return this.m_seq; }

    private double[] calcAcc(ArrayList<ArrayList<Integer>> true_labels, ArrayList<ArrayList<Integer>> pred_labels ){
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
            for(int j = 0; j < true_labels.get(i).size(); j++){
                label_idx_true = true_labels.get(i).get(j);
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

    public void crossValidation(int k, String prefix){
        m_train_label = new ArrayList<>();
        m_train_string = new ArrayList<>();
        m_test_label = new ArrayList<>();
        m_test_string = new ArrayList<>();

        double[][] acc = new double[k][3];

        int[] masks = new int[m_seq.getStrings().size()];
        Random rand = new Random();
        for(int i = 0; i < masks.length; i++)
            masks[i] = rand.nextInt(k);
        System.out.format("[Info]Start RANDOM cross validation...\n");
        //use this loop to iterate all the folders, set train and test
        for(int i = 0; i < k; i++){
            for(int j = 0; j < masks.length; j++){
                if(masks[j] == i){
                    m_test_string.add(m_seq.getStrings().get(j));
                    m_test_label.add(m_seq.getLabels().get(j));
                }else{
                    m_train_string.add(m_seq.getStrings().get(j));
                    m_train_label.add(m_seq.getLabels().get(j));
                }
            }

            System.out.format("==========\n[Info]Fold No. %d: train size = %d, test size = %d...\n",
                    i, m_train_string.size(), m_test_string.size());

            // Create customized training data and testing data (String4Learning).
            ArrayList<String4Learning> training_data = m_seq.string4Learning(m_train_string, m_train_label);
            ArrayList<String4Learning> testing_data = m_seq.string4Learning(m_test_string, null);

            // Build up a graph learner and train it using training data.
            GraphLearner m_graphLearner = new GraphLearner(training_data);

            // Train
            long start = System.currentTimeMillis();
            ArrayList<ArrayList<Integer>> trainPrediction = m_graphLearner.doTraining(30);
            double acc_cur = calcAcc(m_train_label, trainPrediction)[0];
            System.out.format("cur train acc: %f\n", acc_cur);

            m_graphLearner.SaveWeights(String.format("%s/weights.txt", prefix));

            // Apply the trained model to the test set.
            ArrayList<FactorGraph> testGraphSet = m_graphLearner.buildFactorGraphs_test(testing_data);
            ArrayList<ArrayList<Integer>> testPrediction = m_graphLearner.doTesting(testGraphSet);
            acc[i] = calcAcc(m_test_label, testPrediction);

            System.out.format("[Stat]Train/test finished in %.2f seconds: acc_all = %.2f, acc_phrase = %.2f, acc_out = %.2f\n",
                    (System.currentTimeMillis()-start)/1000.0, acc[i][0], acc[i][1], acc[i][2]);
            m_train_string.clear();
            m_train_label.clear();
            m_test_string.clear();
            m_test_label.clear();
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

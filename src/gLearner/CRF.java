package gLearner;

//import edu.umass.cs.mallet.grmm.types.*;
import cc.mallet.grmm.types.Factor;
import cc.mallet.grmm.types.FactorGraph;
import gLearner.GraphLearner;
import gLearner.SeqAnalyzer;
import gLearner.String4Learning;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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



    public void activeLearning(ArrayList<Sequence> sequences, int train_k, int candi_k, int test_k,
                               String prefix, int maxIter, int tuple_k, int budget_k, String model){
        //get train and test index
//        m_seq.genTrainTestIdx(prefix, train_k, test_k);
//        ArrayList<Integer> train_idx = m_seq.loadIdx(prefix, "train", train_k);
//        ArrayList<Integer> test_idx = m_seq.loadIdx(prefix, "test", test_k);
//        ArrayList<Integer> candidate_idx = m_seq.loadIdx(prefix, "candidate",
//                m_seq.getStrings().size() - train_k - test_k);
        System.out.format("CRF START active learning with tuple=%d, budget=%d, model=%s...\n", tuple_k, budget_k, model);

        ArrayList<Sequence> train_seqs = new ArrayList<>(sequences.subList(0, train_k));
        ArrayList<Sequence> candi_seqs = new ArrayList<>(sequences.subList(train_k, train_k + candi_k));
        ArrayList<Sequence> test_seqs = new ArrayList<>(sequences.subList(sequences.size() - test_k, sequences.size()));

        //initial train and test
        Map<Integer, Double> weights = new TreeMap<>();

        ArrayList<int[]> train_label = new ArrayList<>();
        ArrayList<int[]> pred_label = new ArrayList<>();
        ArrayList<String4Learning> training_data = new ArrayList<>();
        ArrayList<Sequence> trace_samples = new ArrayList<>();

//        for(int i = 0; i < train_idx.size(); i++){
//            train_label.add(m_seq.getLabels().get(train_idx.get(i)));
//            pred_label.add(m_seq.getLabels().get(train_idx.get(i)));
//            training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(train_idx.get(i)), "train", weights));
//            trace_samples.add(m_seq.getSequences().get(train_idx.get(i)));
//        }
//
//        ArrayList<int[]> test_label = new ArrayList<>();
//        ArrayList<Sequence> testing_seq = new ArrayList<>();
//        for(int i = 0; i < test_idx.size(); i++){
//            test_label.add(m_seq.getLabels().get(test_idx.get(i)));
//            testing_seq.add(m_seq.getSequences().get(test_idx.get(i)));
//        }

        for(Sequence seq : train_seqs){
            train_label.add(seq.getLabelIDs());
            pred_label.add(seq.getLabelIDs());
            training_data.add(m_seq.getStr4Learning(seq, "train", weights));
            trace_samples.add(seq);
        }

        ArrayList<int[]> test_label = new ArrayList<>();
        ArrayList<Sequence> testing_seq = new ArrayList<>();
        for(Sequence seq : test_seqs){
            test_label.add(seq.getLabelIDs());
            testing_seq.add(seq);
        }

        // to record results
        ArrayList<Integer> costs = new ArrayList<>();
        ArrayList<Double> acc_all_list = new ArrayList<>();
        ArrayList<Double> acc_phrase_list = new ArrayList<>();
        ArrayList<Double> acc_out_list = new ArrayList<>();
        ArrayList<String[]> tokens = new ArrayList<>();
        ArrayList<String[]> labels = new ArrayList<>();
        ArrayList<String[]> pred1_labels = new ArrayList<>();

        //active learning
        GraphLearner m_graphLearner = null;
        double TP = 0, TPFN = 0, TPFP = 0, TP_accumulate = 0, TPFP_accumulate = 0, TPFN_accumulate = 0;
        for(int count = 14 * train_k; count <= budget_k; count += tuple_k){


            System.out.format("==========\n[Info]Cost %d in %d bugdet...\n", count, budget_k);

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
            for (int l = 0; l < testing_seq.size(); l++) {
                Sequence seq = testing_seq.get(l);
                testGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));
                pred_tmp = m_graphLearner.doTesting(testGraph);
                testPrediction.add(pred_tmp);
            }
            double[] acc = calcAcc(test_label, testPrediction);

            System.out.format("[Stat]Train/test finished in %.2f seconds: acc_all = %.2f, acc_phrase = %.2f, acc_out = %.2f\n",
                    (System.currentTimeMillis() - start) / 1000.0, acc[0], acc[1], acc[2]);
            acc_all_list.add(acc[0]);
            acc_phrase_list.add(acc[1]);
            acc_out_list.add(acc[2]);
            costs.add(count);

            TP_accumulate += TP;
            TPFP_accumulate += TPFP;
            TPFN_accumulate += TPFN;
            double prec_current = TPFP > 0? TP/TPFP : -1;
            double prec_accumulate = TPFP_accumulate > 0? TP_accumulate/TPFP_accumulate : -1;
            double recall_current = TPFN > 0? TP/TPFN : -1;
            double recall_accumulate = TPFN_accumulate > 0? TP_accumulate / TPFN_accumulate : -1;
            System.out.format("[Stat]hitting rate: current_precision = %f, accumulate_precision = %f, " +
                            "current_recall = %f, accumulate_recall = %f\n",
                    prec_current, prec_accumulate, recall_current, recall_accumulate);
            TP = 0;
            TPFP = 0;
            TPFN = 0;

            // begin query
            if(tuple_k == 14 && model.equals("random")){//choose a random whole sequence
                Random r = new Random();
                int random_j = r.nextInt(candi_seqs.size());
                train_label.add(candi_seqs.get(random_j).getLabelIDs());
                training_data.add(m_seq.getStr4Learning(candi_seqs.get(random_j), "train", weights));
                candi_seqs.remove(random_j);
                trace_samples.add(candi_seqs.get(random_j));
            } else if(tuple_k == 14 && model.equals("confidence")){//choose whole sequence with minimum confidence
                double min = Double.MAX_VALUE;
                int uncertain_j = 0;
                FactorGraph tmpGraph, targetGraph = new FactorGraph();
                double tmpConficence;
                ArrayList<Double> confis = new ArrayList<>();
                ArrayList<FactorGraph> graphs = new ArrayList<>();
                ArrayList<Integer> targetPred = new ArrayList<>();

                for(int j = 0; j < candi_seqs.size(); j++){
                    Sequence seq = candi_seqs.get(j);
                    tmpGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));
                    tmpConficence = m_graphLearner.calcConfidence(tmpGraph);
                    if(tmpConficence < min){
                        min = tmpConficence;
                        uncertain_j = j;
                        targetGraph = tmpGraph;
                    }

                    confis.add(tmpConficence);
                    graphs.add(tmpGraph);
                }

                System.out.format("seq's confidence: %f\n", min);

                targetPred = m_graphLearner.doTesting(targetGraph);
                System.out.println("pred: " + Arrays.toString(targetPred.toArray()));

                //use model's prediction with true subsequence
                int[] true_label = candi_seqs.get(uncertain_j).getLabelIDs();
                System.out.println("true: " + Arrays.toString(true_label));

                train_label.add(candi_seqs.get(uncertain_j).getLabelIDs());
                training_data.add(m_seq.getStr4Learning(candi_seqs.get(uncertain_j), "train", weights));
                candi_seqs.remove(uncertain_j);
                trace_samples.add(candi_seqs.get(uncertain_j));
            } else {//choose sub sequence
                double min = Double.MAX_VALUE;
                int uncertain_j = 0, uncertain_k=0;
                FactorGraph tmpGraph, targetGraph = new FactorGraph();
                double[] tuple_confidence;
                double tmpConficence;
                ArrayList<Integer> tmpPred = new ArrayList<>();
                ArrayList<Integer> targetPred = new ArrayList<>();

                // first find the sequence with least confidence
                for(int j = 0; j < candi_seqs.size(); j++) {
                    Sequence seq = candi_seqs.get(j);
                    tmpGraph = m_graphLearner.buildFactorGraphs_test(m_seq.getStr4Learning(seq, "test", weights));

                    tmpConficence = m_graphLearner.calcConfidence(tmpGraph);
                    if (tmpConficence < min) {
                        min = tmpConficence;
                        uncertain_j = j;
                        targetGraph = tmpGraph;
                    }
                }

                System.out.format("seq's confidence: %f\n", min);

                targetPred.clear();
                tuple_confidence = m_graphLearner.calcTupleUncertainty(targetGraph, targetPred, tuple_k, model);

//                min = Double.MAX_VALUE;
//                for(int k = 0; k < tuple_confidence.length; k++){
//                    if(tuple_confidence[k] < min){
//                        min = tuple_confidence[k];
//                        uncertain_k = k;
//                    }
//                }
//                System.out.format("position: [%d, %d), confidence: %f\n",
//                        uncertain_k, uncertain_k + tuple_k, min);
                System.out.format("%s: %s\n", model, Arrays.toString(tuple_confidence));

                //sort the confidence
                TreeMap<Double,Integer> confidence_map = new TreeMap<Double,Integer>();
                for(int j = 0;j < tuple_confidence.length; j++ ) {
                    confidence_map.put(tuple_confidence[j], j);
                }

                if(model.equals("random")) {
                    confidence_map.clear();
                    ArrayList<Integer> tmp = new ArrayList<>();
                    for(int j = 0;j < tuple_confidence.length; j++ ) {
                        tmp.add(j);
                    }
                    Collections.shuffle(tmp);
                    for(int j = 0;j < tuple_confidence.length; j++ ) {
                        confidence_map.put((double)j, tmp.get(j));
                    }
                }

                System.out.println("sort: " + confidence_map.keySet().toString());
                System.out.println("indx: " + confidence_map.values().toString());

                //get all the positions
                Set<Integer> pos = new HashSet<>();
                for(int j = 0; j < budget_k; j++){
                    int idx = (int) confidence_map.values().toArray()[j];
                    for(int k = idx; k < idx + tuple_k; k++){
                        pos.add(k);
                    }
                }
                System.out.format("budget %d: %s\n", budget_k, pos.toString());

                System.out.println("pred: " + Arrays.toString(targetPred.toArray()));

                //use model's prediction with true subsequence
                int[] true_label = candi_seqs.get(uncertain_j).getLabelIDs();
                System.out.println("true: " + Arrays.toString(true_label));

                int[] query_label = new int[targetPred.size()];
                int[] predict_label = new int[targetPred.size()];
                for(int j = 0; j < targetPred.size(); j++) {
                    query_label[j] = targetPred.get(j);
                    predict_label[j] = targetPred.get(j);
                    if(targetPred.get(j) != true_label[j]){
                        TPFN += 1;
                    }
                }
                TPFP += pos.size();
                for(Integer idx : pos){
                    query_label[idx] = true_label[idx];
                    if(targetPred.get(idx) != true_label[idx])
                        TP += 1;
                }
                System.out.println("quer: " + Arrays.toString(query_label));

                train_label.add(true_label);
                pred_label.add(predict_label);
                Sequence query_seq = candi_seqs.get(uncertain_j);
                query_seq.setLabelIDs(query_label);
                System.out.println("2trn: " + Arrays.toString(query_seq.getLabelIDs()));
                training_data.add(m_seq.getStr4Learning(query_seq, "train", weights));

//                train_label.add(Arrays.copyOfRange(m_seq.getLabels().get(candidate_idx.get(uncertain_j)), uncertain_k, tuple_k));
//                training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(candidate_idx.get(uncertain_j)).getSubSeq(uncertain_k, tuple_k),
//                        "train", weights));

                //use only true subsequence
                candi_seqs.remove(uncertain_j);
                trace_samples.add(query_seq);
            }
        }

        //output result
        prefix = String.format("%s_CRF_train%d_tuple%d_%s", prefix, train_k, tuple_k, model);
        File acc_all_file = new File(String.format("%s_all.txt", prefix));
        File acc_phrase_file = new File(String.format("%s_phrase.txt", prefix));
        File acc_out_file = new File(String.format("%s_out.txt", prefix));
        File check_file = new File(String.format("%s_check.txt", prefix));
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(acc_all_file));
            //print indexes
            writer.write("cost,acc\n");
            for(int i = 0; i < costs.size(); i++)
                writer.write(String.format("%d,%f\n", costs.get(i), acc_all_list.get(i)));
            writer.close();

            writer = new BufferedWriter((new FileWriter(acc_phrase_file)));
            writer.write("cost,acc\n");
            for(int i = 0; i < costs.size(); i++)
                writer.write(String.format("%d,%f\n", costs.get(i), acc_phrase_list.get(i)));
            writer.close();

            writer = new BufferedWriter((new FileWriter(acc_out_file)));
            writer.write("cost,acc\n");
            for(int i = 0; i < costs.size(); i++)
                writer.write(String.format("%d,%f\n", costs.get(i), acc_out_list.get(i)));
            writer.close();

//            File checkfile = new File(String.format("%s_check.txt", prefix));
//            writer = new BufferedWriter(new FileWriter(checkfile));
//            for(int i = 0; i < tokens.size(); i++) {
//                writer.write(String.format("%d th query with %d positions:\n", i, tuple_k));
//                writer.write(String.format("tokens: %s\n", Arrays.toString(tokens.get(i))));
//                writer.write(String.format("labels: %s\n", Arrays.toString(labels.get(i))));
//                writer.write(String.format("Pred_1: %s\n", Arrays.toString(pred1_labels.get(i))));
////                writer.write(String.format("Pred_2: %s\n", Arrays.toString(pred2_labels.get(i))));
//                writer.write("\n");
//            }
//            writer.close();

            writer = new BufferedWriter((new FileWriter(check_file)));
            for(int i = train_k; i < trace_samples.size(); i++) {
                writer.write("token: " + trace_samples.get(i).getContent() + "\n");

                writer.write("true: ");
                for(Integer id : train_label.get(i)) {
                    writer.write(String.format("%s,", m_seq.getLabelName(id)));
                }
                writer.write("\n");

                writer.write("pred: ");
                for(Integer id : pred_label.get(i)) {
                    writer.write(String.format("%s,", m_seq.getLabelName(id)));
                }
                writer.write("\n");

                writer.write("quer: ");
                for(Integer id : trace_samples.get(i).getLabelIDs()) {
                    writer.write(String.format("%s,", m_seq.getLabelName(id)));
                }
                writer.write("\n");

                writer.write("\n");
            }
            writer.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void oneFold(int train_end, int test_start, int maxIter) {
        double[] acc = new double[3];

        ArrayList<int[]> train_label = new ArrayList<>();
        ArrayList<String4Learning> training_data = new ArrayList<>();
        ArrayList<String4Learning> tmp_data = new ArrayList<>();

        ArrayList<int[]> test_label = new ArrayList<>();
        ArrayList<Sequence> testing_seq = new ArrayList<>();


        Map<Integer, Double> weights = new TreeMap<>();

        for(int i = 0; i < m_seq.getStrings().size(); i++) {
            if(i > test_start) {
                test_label.add(m_seq.getLabels().get(i));
                testing_seq.add(m_seq.getSequences().get(i));
            } else if (i < train_end) {
                train_label.add(m_seq.getLabels().get(i));
                training_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(i), "train", weights));
            }
        }

        System.out.format("total size = %d, train size = %d, test size = %d...\n",
                m_seq.getStrings().size(), training_data.size(), testing_seq.size());

        // Build up a graph learner and train it using training data.
        GraphLearner m_graphLearner = new GraphLearner(training_data);

        // Train
        long start = System.currentTimeMillis();
        ArrayList<ArrayList<Integer>> trainPrediction = m_graphLearner.doTraining(maxIter);
        trainPrediction.clear();

//        m_graphLearner.SaveWeights(String.format("%s/weights.txt", prefix));
        weights = m_graphLearner.getWeights();

        for(int i = 0; i < m_seq.getSequences().size(); i++) {
            if (i < train_end) {
                tmp_data.add(m_seq.getStr4Learning(m_seq.getSequences().get(i), "test", weights));
            }
        }

        for(int l = 0; l < tmp_data.size(); l++) {
            FactorGraph tmpGraph = m_graphLearner.buildFactorGraphs_test(tmp_data.get(l));
            trainPrediction.add(m_graphLearner.doTesting(tmpGraph));
        }
        double acc_cur = calcAcc(train_label, trainPrediction)[0];
        System.out.format("cur train acc: %f\n", acc_cur);


        m_graphLearner.SaveWeights("./data/weights.txt");
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
            } else if(acc_tmp[0] > 0.95) {
                System.out.format("===== good =====\n");
                System.out.format("Token: %s\n", seq.getContent());
            } else {
                System.out.format("===== medium =====\n");
                System.out.format("Token: %s\n", seq.getContent());
            }
            if(acc_tmp[0] < 0.96 || acc_tmp[0] > 0.95) {
                System.out.format("True: ");
                int[] labels = seq.getLabelIDs();
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
        acc = calcAcc(test_label, testPrediction);
        System.out.format("[Stat]Train/test finished in %.2f seconds: acc_all = %.4f, acc_phrase = %.4f, acc_out = %.4f\n",
                (System.currentTimeMillis()-start)/1000.0, acc[0], acc[1], acc[2]);
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
                    int[] labels = seq.getLabelIDs();
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

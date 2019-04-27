package gLearner;

import structures.LanguageModel;
import structures.Token;

import java.io.*;
import java.util.*;

public class HMM {
    // structures to store c(ti), c(wi), c(wi,ti) in emission, c(ti,tj) in transition
    HashMap<String, Token> m_tagset_stats;
    HashMap<String, Token> m_tokenset_stats;
    HashMap<String, Token> m_emission_stats;
    HashMap<String, Token> m_transition_stats;

    //this structure is for language modeling
    LanguageModel m_wordTag;
    LanguageModel m_tagTag;

    public HMM() {
        m_tagset_stats = new HashMap<>();
        m_tokenset_stats = new HashMap<>();
        m_emission_stats = new HashMap<>();
        m_transition_stats = new HashMap<>();
    }

    public void clear(){
        m_tagset_stats.clear();
        m_tokenset_stats.clear();
        m_emission_stats.clear();
        m_transition_stats.clear();
    }

    private double test(ArrayList<Sequence> test_seqs){
        // test on testing data and output accuracy
        int tp = 0;
        int fp = 0;
        for(Sequence seq : test_seqs){
            String[] words = seq.getTokens();
            String[] truePOSs = seq.getLabels();

            // predict pos using HMM
            String[] predPOSs = Viterbi4HMM(words);

            for(int n = 0; n < predPOSs.length; n++){
                String predPos = predPOSs[n];
                String truePos = truePOSs[n];
                if(predPos.equals(truePos)){
                    tp += 1;
                }else{
                    fp += 1;
                }
            }
        }
        double overall_accuracy = (double)tp/(tp + fp);
        System.out.format("[Info]Overall accuracy: %f\n", overall_accuracy);
        return overall_accuracy;
    }

    public void activeLearning(ArrayList<Sequence> sequences, int train_k, int candi_k, int test_k, String prefix,
                               int tuple_k, int budget_k, String model) {
//        Collections.shuffle(sequences);
        System.out.format("START active learning with tuple=%d, budget=%d, model=%s...\n", tuple_k, budget_k, model);

        ArrayList<Sequence> train_seqs = new ArrayList<>(sequences.subList(0, train_k));
        ArrayList<Sequence> candi_seqs = new ArrayList<>(sequences.subList(train_k, train_k + candi_k));
        ArrayList<Sequence> test_seqs = new ArrayList<>(sequences.subList(sequences.size() - test_k, sequences.size()));

        ArrayList<Integer> costs = new ArrayList<>();
        ArrayList<Double> accs = new ArrayList<>();
        ArrayList<String[]> tokens = new ArrayList<>();
        ArrayList<String[]> labels = new ArrayList<>();
        ArrayList<String[]> pred1_labels = new ArrayList<>();
        ArrayList<String[]> pred2_labels = new ArrayList<>();

        // train HMM using MLE
        train(train_seqs, 2, 0.3, 0.1);
        costs.add(14 * train_k);
        accs.add(test(test_seqs));

        for(int count = tuple_k; count <= budget_k; count += tuple_k){
            if (model.equals("random")) {
                Random r = new Random();
                int random_j = r.nextInt(candi_seqs.size());
                Sequence new_seq = candi_seqs.get(random_j);
                candi_seqs.remove(random_j);

                if(tuple_k > new_seq.getLength()){
                    tuple_k = new_seq.getLength();
                }
                ArrayList<Integer> arr = new ArrayList<>();
                for (int i = 0; i < new_seq.getLength(); i++) {
                    arr.add(i);
                }
                Collections.shuffle(arr);

                String[] true_labels = new_seq.getLabels();
                String[] pred_labels = Viterbi4HMM(new_seq.getTokens());
                for (int i = 0; i < tuple_k; i++){
                    pred_labels[arr.get(i)] = true_labels[arr.get(i)];
                }


                new_seq.setLabels(pred_labels);
                train_seqs.add(new_seq);
            } else if (model.equals("confidence")) {
                double min = Double.MAX_VALUE;
                int choose_j = 0;
                ArrayList<String[]> NbestPreds = new ArrayList<>();
                for (int i = 0; i < candi_seqs.size(); i++) {
                    double[] NbestProbs = new double[2];
                    ArrayList<String[]> preds = NbestViterbi4HMM(candi_seqs.get(i).getTokens(), NbestProbs);
                    if (min > NbestProbs[0]){
                        min = NbestProbs[0];
                        choose_j = i;
                        NbestPreds = preds;
                    }
                }
                Sequence new_seq = candi_seqs.get(choose_j);
                train_seqs.add(new_seq);
                candi_seqs.remove(choose_j);
                tokens.add(new_seq.getTokens());
                labels.add(new_seq.getLabels());
                pred1_labels.add(NbestPreds.get(0));
                pred2_labels.add(NbestPreds.get(1));
            } else { // margin
                double min = Double.MAX_VALUE;
                int choose_j = 0;
                ArrayList<String[]> NbestPreds = new ArrayList<>();
                for (int i = 0; i < candi_seqs.size(); i++) {
                    double[] NbestProbs = new double[2];
                    ArrayList<String[]> preds = NbestViterbi4HMM(candi_seqs.get(i).getTokens(), NbestProbs);
                    if (min > NbestProbs[0] - NbestProbs[1]){
                        min = NbestProbs[0] - NbestProbs[1];
                        choose_j = i;
                        NbestPreds = preds;
                    }
                }
                Sequence new_seq = candi_seqs.get(choose_j);
                train_seqs.add(new_seq);
                candi_seqs.remove(choose_j);
                tokens.add(new_seq.getTokens());
                labels.add(new_seq.getLabels());
                pred1_labels.add(NbestPreds.get(0));
                pred2_labels.add(NbestPreds.get(1));
            }

            train(train_seqs, 2, 0.3, 0.1);
            costs.add(count);
            accs.add(test(test_seqs));
        }

        System.out.format("file path: %s", prefix);
        try{
            //output result
            prefix = String.format("%s_HMM_train%d_tuple%d_%s", prefix, train_k, tuple_k, model);
            File accfile = new File(String.format("%s_acc.txt", prefix));
            BufferedWriter writer = new BufferedWriter(new FileWriter(accfile));
            writer.write("cost, acc\n");
            for(int i = 0; i < costs.size(); i++)
                writer.write(String.format("%d, %f\n", costs.get(i), accs.get(i)));
            writer.close();

            File checkfile = new File(String.format("%s_check.txt", prefix));
            writer = new BufferedWriter(new FileWriter(checkfile));
            for(int i = 0; i < tokens.size(); i++) {
                writer.write(String.format("%d th query with %d positions:\n", i, tuple_k));
                writer.write(String.format("tokens: %s\n", Arrays.toString(tokens.get(i))));
                writer.write(String.format("labels: %s\n", Arrays.toString(labels.get(i))));
                writer.write(String.format("Pred_1: %s\n", Arrays.toString(pred1_labels.get(i))));
                writer.write(String.format("Pred_2: %s\n", Arrays.toString(pred2_labels.get(i))));
                writer.write("\n");
            }
            writer.close();

            System.out.format("write to file path: %s", prefix);
        } catch (IOException e){
            e.printStackTrace();
        }

    }


    public void train(ArrayList<Sequence> seqs, int Ngram, double lambda, double delta){
        clear();
        /*** collect all statistics ***/
        for(Sequence seq : seqs){
            String[] words = seq.getTokens();
            String[] poses = seq.getLabels();

            for(int i = 0;i < words.length; i++){
                addToken(m_emission_stats, poses[i] + "#" + words[i]);
                addToken(m_tagset_stats, poses[i]);
                addToken(m_tokenset_stats, words[i]);

                String pos = poses[i];
                for (int j = i - 1; j >= Math.max(0, i - 1); j--) {
                    pos = poses[j] + "#" + pos;
                }
                if(pos.contains("#")) {
                    addToken(m_transition_stats, pos);
                }
            }
        }

        /*** MLE using language model ***/
        //word-tag language model for emission probability
        m_wordTag = new LanguageModel(Ngram, lambda, delta);
        m_wordTag.setModel(m_emission_stats);

        // tag-tag language model for transition probability
        m_tagTag = new LanguageModel(Ngram, lambda, delta);
        m_tagTag.setModel(m_transition_stats);

        //set reference model for smoothing
        LanguageModel refer = new LanguageModel(Ngram-1, lambda, delta);
        refer.setModel(m_tagset_stats);
        m_tagTag.setReference(refer);
        m_wordTag.setReference(refer);
    }

    public void addToken(HashMap<String, Token> vector, String token){
        if(vector.containsKey(token)){
            vector.get(token).addTF(1);
        } else{
            Token newTk = new Token(token);
            newTk.setValue(1);
            vector.put(token, newTk);
        }
    }

    public ArrayList<String[]> NbestViterbi4HMM(String[] words, double[] NbestProbs){
        int wordN = words.length;
        int tagN = m_tagset_stats.size();
        ArrayList<String> tagset = new ArrayList<>(m_tagset_stats.keySet());

        ArrayList<String[]> NbestPOS = new ArrayList<>();

        double[][][] trellis = new double[2][tagN][wordN + 1];
        String[][][] position = new String[2][tagN][wordN + 1];

        for(int j = 0;j < tagN; j++){
            trellis[0][j][0] = 1.0 / tagN;
            trellis[1][j][0] = 1.0 / tagN;
        }

        for(int i = 1; i < wordN + 1; i++){
            for(int j = 0; j < tagN; j++){
                int range = 2;
                if (i == 1) {
                    range = 2;
                }
                HashMap<String, Double> probs = new HashMap<>();
                for (int l = 0; l < range; l++) {
                    for (int k = 0; k < tagN; k++) {
                        String key = Integer.toString(l) + "#" + Integer.toString(k);
                        Double value = trellis[l][k][i - 1] * m_tagTag.calcAdditiveSmoothedProb(tagset.get(k)
                                + "#" + tagset.get(j));
                        probs.put(key, value);
                    }
                }
                List<Map.Entry<String, Double>> probList = new ArrayList<>(probs.entrySet());

                Collections.sort(probList, new Comparator<Map.Entry<String, Double>>() {
                    public int compare(Map.Entry<String, Double> o1,
                                       Map.Entry<String, Double> o2) {
                        return o2.getValue().compareTo(o1.getValue());
                    }
                });

                trellis[0][j][i] = m_wordTag.calcAdditiveSmoothedProb(tagset.get(j)
                        + "#" + words[i-1]) * probList.get(0).getValue();
                position[0][j][i] = probList.get(0).getKey();

                trellis[1][j][i] = m_wordTag.calcAdditiveSmoothedProb(tagset.get(j)
                        + "#" + words[i-1]) * probList.get(1).getValue();
                position[1][j][i] = probList.get(1).getKey();
            }
        }

        HashMap<String, Double> probs = new HashMap<>();
        for (int k = 0; k < tagN; k++) {
            for(int l = 0; l < 2; l++){
                String key = Integer.toString(l) + "#" + Integer.toString(k);
                Double value = trellis[l][k][wordN];
                probs.put(key, value);
            }
        }
        List<Map.Entry<String, Double>> probList = new ArrayList<>(probs.entrySet());

        Collections.sort(probList, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> o1,
                               Map.Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });

        int[] top1_tag_seq = new int[wordN + 1];
        int[] top2_tag_seq = new int[wordN + 1];
        int[] top3_tag_seq = new int[wordN + 1];
        int l1 = Integer.valueOf(probList.get(0).getKey().split("#")[0]);
        int k1 = Integer.valueOf(probList.get(0).getKey().split("#")[1]);
        int l2 = Integer.valueOf(probList.get(1).getKey().split("#")[0]);
        int k2 = Integer.valueOf(probList.get(1).getKey().split("#")[1]);
        int l3 = Integer.valueOf(probList.get(2).getKey().split("#")[0]);
        int k3 = Integer.valueOf(probList.get(2).getKey().split("#")[1]);
        top1_tag_seq[wordN] = k1;
        top2_tag_seq[wordN] = k2;
        top3_tag_seq[wordN] = k3;
        for(int i = wordN; i > 0; i--){
            String[] pos1 = position[l1][top1_tag_seq[i]][i].split("#");
            String[] pos2 = position[l2][top2_tag_seq[i]][i].split("#");
            String[] pos3 = position[l3][top3_tag_seq[i]][i].split("#");
            top1_tag_seq[i - 1] = Integer.valueOf(pos1[1]);
            top2_tag_seq[i - 1] = Integer.valueOf(pos2[1]);
            top3_tag_seq[i - 1] = Integer.valueOf(pos3[1]);
            l1 = Integer.valueOf(pos1[0]);
            l2 = Integer.valueOf(pos2[0]);
            l3 = Integer.valueOf(pos3[0]);
        }

        int flag = 1;//1 indicate that top1 == top2
//        for(int i = 0; i < wordN; i++){
//            if(top1_tag_seq[i+1] != top2_tag_seq[i+1]){
//                flag = 0;
//                System.out.format("[Warning]Same top1 and top2!\n");
//                break;
//            }
//        }
        if (flag == 1){
            for(int i = 0; i < wordN+1; i++){
                top2_tag_seq[i] = top3_tag_seq[i];
            }
        }


        String[] top1POS = new String[wordN];
        String[] top2POS = new String[wordN];
        for(int i = 0; i < wordN; i++){
            top1POS[i] = tagset.get(top1_tag_seq[i + 1]);
            top2POS[i] = tagset.get(top2_tag_seq[i + 1]);
        }
        NbestPOS.add(top1POS);
        NbestPOS.add(top2POS);

        NbestProbs[0] = probList.get(0).getValue();
        NbestProbs[1] = probList.get(1).getValue();
        if (flag == 1){
            NbestProbs[1] = probList.get(2).getValue();
        }


        return NbestPOS;
    }

    public String[] Viterbi4HMM(String[] words){
        int wordN = words.length;
        int tagN = m_tagset_stats.size();
        ArrayList<String> tagset = new ArrayList<>(m_tagset_stats.keySet());

        String[] predPOS = new String[wordN];
        double[][] trellis = new double[tagN][wordN + 1];
        int[][] position = new int[tagN][wordN + 1];

        for(int j = 0;j < tagN; j++){
            trellis[j][0] = 1.0 / tagN;
        }

        for(int i = 1; i < wordN + 1; i++){
            for(int j = 0; j < tagN; j++){
                double maxPre = 0.0;
                for(int k = 0; k < tagN; k++){
                    double pre = trellis[k][i-1] *
                            m_tagTag.calcAdditiveSmoothedProb(tagset.get(k)
                                    + "#" + tagset.get(j));
                    if(pre > maxPre){
                        maxPre = pre;
                        position[j][i] = k;
                    }
                }

                trellis[j][i] = m_wordTag.calcAdditiveSmoothedProb(tagset.get(j)
                        + "#" + words[i-1]) * maxPre;
            }
        }

        double maxProb = 0.0;
        int[] tag_seq = new int[wordN + 1];
        for(int j = 0; j < tagN; j++){
            if(trellis[j][wordN] > maxProb){
                maxProb = trellis[j][wordN];
                tag_seq[wordN] = j;
            }
        }

        for(int i = wordN; i > 0; i--){
            tag_seq[i - 1] = position[tag_seq[i]][i];
        }

        for(int i = 0; i < wordN; i++){
            predPOS[i] = tagset.get(tag_seq[i + 1]);
        }

        return predPOS;
    }

    // sampling word and tag using learned HMM
    public String sampling(String pre, Collection<String> candidates, LanguageModel lm){
        while(true) {
            double prob = Math.random();
            double cumulateProb = 0.0;
            for (String cur : candidates) {
                String token = pre + "#" + cur;
                double curProb = 0.0;
                curProb = lm.calcAdditiveSmoothedProb(token);
                cumulateProb += curProb;
                if ((prob - cumulateProb) <= 0) {
                    return token.substring(token.lastIndexOf('#') + 1, token.length());
                }
            }
        }
    }

    public void genSentences(int stnsNum, int stnsLen, String filePath) {
        List<List<String>> stss = new ArrayList<>();
        List<List<String>> postags = new ArrayList<>();
        HashMap<String, Double> likelihood = new HashMap<>();
        for (int i = 0; i < stnsNum; i++){
            List<String> sts = new ArrayList<>();
            List<String> tags = new ArrayList<>();
            double lkl = 0.0;
            String curTg = "START";
            String curWd = "";
            for(int j = 0 ;j < stnsLen; j++){
                // sample tag
                String oldTg = curTg;
                curTg = sampling(oldTg, m_tagset_stats.keySet(), m_tagTag);

                // sample word
                curWd = sampling(curTg, m_tokenset_stats.keySet(), m_wordTag);

                sts.add(curWd);
                tags.add(curTg);
                lkl += Math.log(m_tagTag.calcAdditiveSmoothedProb(oldTg + "#" + curTg));
                lkl += Math.log(m_wordTag.calcAdditiveSmoothedProb(curTg + "#" + curWd));
            }
            stss.add(sts);
            postags.add(tags);
            likelihood.put(String.valueOf(stss.size()-1), lkl);
            System.out.format("-- %d's sentence likelihood: %f\n", i, lkl);
        }

        List<Map.Entry<String, Double>> probRank = sortProb(likelihood);
        Writer writer = null;
        try{
            writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(filePath), "utf-8"));
            for(Map.Entry<String, Double> entry : probRank){
                writer.write(entry.getValue() + "\n");
                for(String str:stss.get(Integer.valueOf(entry.getKey()))){
                    writer.write(str + " ");
                }
                writer.write("\n");
            }
        } catch (IOException e){
            e.printStackTrace();
        } finally {
            try {writer.close();} catch (Exception ex) {/*ignore*/}
        }
    }

    public List<Map.Entry<String, Double>> sortProb(HashMap<String, Double> prob){
        List<Map.Entry<String, Double>> probRank =
                new ArrayList<Map.Entry<String, Double>>(prob.entrySet());

        Collections.sort(probRank, new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });
        return probRank;
    }
}


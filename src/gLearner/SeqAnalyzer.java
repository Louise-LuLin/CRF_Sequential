package gLearner;

import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.Variable;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SeqAnalyzer {
    private String m_source;
    private ArrayList<String> m_labelNames;
    private HashMap<String, Integer> m_labelNameIndex;

    private ArrayList<String> m_tokenNames;
    private HashMap<String, Integer> m_tokenNameIndex;

    private ArrayList<String> m_strList;
    private ArrayList<ArrayList<Integer>> m_labelList;
    private ArrayList<ArrayList<Integer>> m_tokenList;

    private Map<Integer, Boolean> m_mask;

    public SeqAnalyzer(String source){
        this.m_source = source;
        m_labelNames = new ArrayList<>();
        m_labelNameIndex = new HashMap<>();
        m_tokenNames = new ArrayList<>();
        m_tokenNameIndex = new HashMap<>();
        m_strList = new ArrayList<>();
        m_labelList = new ArrayList<>();
        m_tokenList = new ArrayList<>();
        m_mask = null;
    }

    public void setMask(Map<Integer, Boolean> masks) { m_mask = masks; }

    public ArrayList<String> getLabelNames(){return this.m_labelNames; }

    public HashMap<String, Integer> getLabelNameIndex(){ return this.m_labelNameIndex; }

    public ArrayList<String> getStrings(){ return this.m_strList; }

    public ArrayList<ArrayList<Integer>> getLabels() { return this.m_labelList; }

    public ArrayList<ArrayList<Integer>> getTokens() { return this.m_tokenList; }

    public void saveTokenNames(String filePath){
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filePath)));
            for(int i = 0; i < m_tokenNames.size(); i++)
                writer.write(i + "\t" + m_tokenNames.get(i) + "\n");
            writer.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void saveLabelNames(String filePath){
        try{
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(filePath)));
            for(int i = 0; i < m_labelNames.size(); i++)
                writer.write(i + "\t" + m_labelNames.get(i) + "\n");
            writer.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public void loadString(String filePath, int maxNum){
        m_strList.clear();
        m_tokenList.clear();
        // Read training strings.
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            ArrayList<Integer> token_idxs;
            while ((line = br.readLine()) != null && line.length() > 0) {
                token_idxs = new ArrayList<>();
                int idx;
                for(int i = 0 ; i < line.length(); i++){
                    idx = getTokenIndex(Character.toString(line.charAt(i))); //dynamically expand tokenNames: each char to string
                    token_idxs.add(idx);
                }
                m_tokenList.add(token_idxs);
                m_strList.add(line);
                if(m_strList.size() > maxNum)
                    break;
            }
        } catch (Exception e){
            System.err.format("[Err]File %s doesn't exist.\n", filePath);
        }
    }

    public void loadLabel(String filePath, int maxNum){
        m_labelList.clear();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            String[] labels;
            ArrayList<Integer> label_idxs;
            while ((line = br.readLine()) != null && line.length() > 0) {
                labels = line.split(",");
                label_idxs = new ArrayList<>();
                int idx;
                for(String s: labels){
                    if(s.equals(""))
                        continue;
                    idx = getLabelIndex(s); //dynamically expand labelNames
                    label_idxs.add(idx);
                }
                m_labelList.add(label_idxs);
                if(m_labelList.size() > maxNum)
                    break;
            }
        } catch (Exception e){
            System.err.format("[Err]File %s doesn't exist.\n", filePath);
        }
    }

    private ArrayList<ArrayList<Integer>> string2vec(ArrayList<String> strings){
        ArrayList<ArrayList<Integer>> token_vec = new ArrayList<>();
        ArrayList<Integer> token_idxs;
        for(String str : strings){
            token_idxs = new ArrayList<>();
            int idx;
            for(int i = 0 ; i < str.length(); i++){
                idx = getTokenIndex(Character.toString(str.charAt(i))); //dynamically expand tokenNames: each char to string
                token_idxs.add(idx);
            }
            token_vec.add(token_idxs);
        }

        return token_vec;
    }

    public ArrayList<String4Learning> string4Learning(ArrayList<String> strings,
                                                      ArrayList<ArrayList<Integer>> label_vec){

        System.out.format("[Info]Label size: %d, token size: %d\n", m_labelNames.size(), m_tokenNames.size());
        ArrayList<ArrayList<Integer>> token_vec = string2vec(strings);

        // Each string is stored as an object specifying features as table factors.
        ArrayList<String4Learning> slist = new ArrayList<>();

        // Feature vectors. Key: feature type indices. The vectors are of
        // the same length equivalent to the string length.
        HashMap<Integer, ArrayList<Double>> node_features;


        String4Learning str;

        // For each training sample, construct a factor graph, and a list of table factors to specify edge
        // and node features.
        for(int idx_sample = 0; idx_sample < token_vec.size(); idx_sample++){
//            System.out.format("[Info]Constructing %d-th sample\n", idx_sample);

            //step 1: construct the graph
            int varNodeSize = token_vec.get(idx_sample).size();
            Variable[] allVars = new Variable[varNodeSize];
            for(int i = 0; i < allVars.length; i++)
                allVars[i] = new Variable(m_labelNames.size()); //each label variable has this many outcomes
            ArrayList<Factor> factorList = new ArrayList<>();   // list of table factors for the current string
            ArrayList<Integer> featureType = new ArrayList<>(); // corresponding feature ID for each list of factors

            //step 2: add features
            node_features = constructNodeFeature(token_vec.get(idx_sample));
            ArrayList<Double> cur_feature;
            Factor ptl;
            double[] feature_value_arr = new double[m_labelNames.size()];
            for(int j = 0; j < varNodeSize; j++) {// for each node/variable

                // note features:
                // token itself:   0 ~ label_size * (token_size + 2) * (num of type%2==0)
                // token is digit: label_size * (token_size + 2) * 5 (=A) ~ A + label_size * (num of type%2==1)
                for (Integer type : node_features.keySet()) {// for each feature type
                    //skip feature type with mask
                    if(m_mask != null && m_mask.containsKey(type)
                            && m_mask.get(type).booleanValue() == false)
                        continue;

                    cur_feature = node_features.get(type);

                    if (type % 2 == 0) { // x_t/t-1...: type=0,2,4,6,8
                        for (int k = 0; k < m_tokenNames.size(); k++) {
                            for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
                                if (cur_feature.get(j).intValue() == k) {
                                    Arrays.fill(feature_value_arr, 1.0);
                                    feature_value_arr[label_i] = Math.exp(1.0);
                                }
                                else
                                    Arrays.fill(feature_value_arr, 1.0);
                                ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                                factorList.add(ptl);
                                featureType.add((m_tokenNames.size() * m_labelNames.size()) * (type / 2) +
                                        m_labelNames.size() * k + label_i);
                            }
                        }
                    } else { // is digit: type=1,3,5,7,9
                        for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
                            Arrays.fill(feature_value_arr, 1.0);
                            feature_value_arr[label_i] = Math.exp(cur_feature.get(j));
                            ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                            factorList.add(ptl);
                            featureType.add((m_tokenNames.size() * m_labelNames.size()) * 5 +
                                    m_labelNames.size() * (type / 2) + label_i);
                        }
                    }
                }
            }

            // Add all first-order transition features f(y_(i-1),y_i).
//            double[] trans_feature_arr;
//            for(int i=0; i<num_label; i++){
//                for(int j=0; j<num_label; j++){
//                    trans_feature_arr = m_seq.label_transition(i,j);
//                    //System.out.println(trans_feature_arr.toString());
//                    for(int k=0; k<len_string.get(idx_sample)-1; k++){
//                        Factor ptl = LogTableFactor.makeFromLogValues(
//                                new Variable[] {allVars[k], allVars[k+1]}, trans_feature_arr);
////                        Factor ptl = new TableFactor(
////                                new Variable[] {allVars[k], allVars[k+1]}, trans_feature_arr);
//                        factorList.add(ptl);
//                        featureType.add(num_node_feature_type+i*num_label+j);
//                    }
//                }
//            }

            // Add the list of table factors into the sample object.
            if(label_vec != null) {
                str = new String4Learning(factorList, featureType, label_vec.get(idx_sample));
            }else{
                str = new String4Learning(factorList, featureType);
            }
            slist.add(str);
        }

        return slist;
    }

    // build the node features
    public HashMap<Integer,ArrayList<Double>> constructNodeFeature(ArrayList<Integer> sample){
        //0: x_t = o
        //1: x_t is digit
        //2: x_t-1 = o
        //3: x_t-1 is digit
        //4: x_t+1 = o
        //5: x_t+1 is digit
        //6: x_t-2 = o
        //7: x_t-2 is digit
        //8: x_t+2 = o
        //9: x_t+2 is digit
        HashMap<Integer, ArrayList<Double>> node_features = new HashMap<>();

        //0: index of x_t
        ArrayList<Double> x_t = new ArrayList<>();
        //1: if x_t is a digit
        ArrayList<Double> x_t_is_digit = new ArrayList<>();

        //2: if x_t-1 = o
        ArrayList<Double> x_t_pre_1 = new ArrayList<>();
        //3: if x_t-1 is a digit
        ArrayList<Double> x_t_pre_1_is_digit = new ArrayList<>();

        //4: if x_t+1 = o
        ArrayList<Double> x_t_next_1 = new ArrayList<>();
        //5: if x_t+1 is a digit
        ArrayList<Double> x_t_next_1_is_digit = new ArrayList<>();

        //6: if x_t-2 = o
        ArrayList<Double> x_t_pre_2 = new ArrayList<>();
        //7: if x_t-2 is a digit
        ArrayList<Double> x_t_pre_2_is_digit = new ArrayList<>();

        //8: if x_t+2 = o
        ArrayList<Double> x_t_next_2 = new ArrayList<>();
        //9: if x_t+2 is a digit
        ArrayList<Double> x_t_next_2_is_digit = new ArrayList<>();

        int curIdx;
        String curToken;
        for(int i = 0;i < sample.size(); i++){
            //x_t
            curIdx = sample.get(i);
            curToken = m_tokenNames.get(curIdx);
            x_t.add((double) curIdx);
            //x_t is digit
            if(isDigit(curToken))
                x_t_is_digit.add(1.0);
            else
                x_t_is_digit.add(0.0);

            //x_t-1
            if(i == 0)
                curIdx = getTokenIndex("START");
            else
                curIdx = sample.get(i-1);
            curToken = m_tokenNames.get(curIdx);
            x_t_pre_1.add((double) curIdx);
            //x_t-1 is digit
            if(isDigit(curToken))
                x_t_pre_1_is_digit.add(1.0);
            else
                x_t_pre_1_is_digit.add(0.0);

            //x_t-2
            if(i == 0 || i == 1)
                curIdx = getTokenIndex("START");
            else
                curIdx = sample.get(i-2);
            curToken = m_tokenNames.get(curIdx);
            x_t_pre_2.add((double) curIdx);
            //x_t-2 is digit
            if(isDigit(curToken))
                x_t_pre_2_is_digit.add(1.0);
            else
                x_t_pre_2_is_digit.add(0.0);

            //x_t+1
            if(i == sample.size()-1)
                curIdx = getTokenIndex("END");
            else
                curIdx = sample.get(i+1);
            curToken = m_tokenNames.get(curIdx);
            x_t_next_1.add((double) curIdx);
            //x_t+1 is digit
            if(isDigit(curToken))
                x_t_next_1_is_digit.add(1.0);
            else
                x_t_next_1_is_digit.add(0.0);

            //x_t+2
            if(i == sample.size()-1 || i == sample.size()-2)
                curIdx = getTokenIndex("END");
            else
                curIdx = sample.get(i+2);
            curToken = m_tokenNames.get(curIdx);
            x_t_next_2.add((double) curIdx);
            //x_t+2 is digit
            if(isDigit(curToken))
                x_t_next_2_is_digit.add(1.0);
            else
                x_t_next_2_is_digit.add(0.0);

        }

        node_features.put(1,x_t_is_digit);
        node_features.put(3,x_t_pre_1_is_digit);
        node_features.put(5,x_t_next_1_is_digit);
        node_features.put(7,x_t_pre_2_is_digit);
        node_features.put(9,x_t_next_2_is_digit);

        node_features.put(0,x_t);
        node_features.put(2,x_t_pre_1);
        node_features.put(4,x_t_next_1);
        node_features.put(6,x_t_pre_2);
        node_features.put(8,x_t_next_2);

        return node_features;
    }

    // This is the first-order edge feature enumerating all possible label transitions.
    public double[] label_transition(int label1, int label2){
        int len_arr = m_labelNames.size() * m_labelNames.size();
        double[] arr = new double[len_arr];
        Arrays.fill(arr, 0.0);
        arr[label1*m_labelNames.size()+label2] = 1.0;
        return arr;
    }

    // Check if a character (token) is a digit.
    private boolean isDigit(String token){
        char token_char = token.charAt(0);
        return Character.isDigit(token_char);
    }

    //get label index and dynamically expand vocabulary if not found
    private int getLabelIndex(String label){
        if(!m_labelNameIndex.containsKey(label)) {
            m_labelNameIndex.put(label, m_labelNames.size());
            m_labelNames.add(label);
        }
        return m_labelNameIndex.get(label);
    }

    //get token index and dynamically expand token list if not found
    private int getTokenIndex(String token){
        if(!m_tokenNameIndex.containsKey(token)) {
            m_tokenNameIndex.put(token, m_tokenNames.size());
            m_tokenNames.add(token);
        }
        return m_tokenNameIndex.get(token);
    }

}

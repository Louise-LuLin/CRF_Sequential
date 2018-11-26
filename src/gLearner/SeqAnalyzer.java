package gLearner;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import edu.umass.cs.mallet.base.types.Matrix;
import edu.umass.cs.mallet.base.types.SparseMatrixn;
import edu.umass.cs.mallet.grmm.types.Factor;
import edu.umass.cs.mallet.grmm.types.LogTableFactor;
import edu.umass.cs.mallet.grmm.types.Variable;

public class SeqAnalyzer {
    private String m_source;
    
    private ArrayList<String> m_labelNames;//text content of the labels Y
    private HashMap<String, Integer> m_labelNameIndex;//index of the labels Y

    private ArrayList<String> m_tokenNames;//text content of the words X
    private HashMap<String, Integer> m_tokenNameIndex;//index of the words X

    //why do not we create a structure for each input string (e.g., sentence or point name)
    private ArrayList<Sequence> m_seqList;

    private Map<Integer, Boolean> m_mask;

    public SeqAnalyzer(String source){
        this.m_source = source;
        
        m_labelNames = new ArrayList<String>();
        m_labelNameIndex = new HashMap<String, Integer>();
        
        m_tokenNames = new ArrayList<String>();
        m_tokenNameIndex = new HashMap<String, Integer>();
        
        m_seqList = new ArrayList<Sequence>();
        m_mask = null;
    }

    public String getSource() { return m_source; } //where we load the data

    public void setMask(Map<Integer, Boolean> masks) { m_mask = masks; }

    public ArrayList<String> getLabelNames(){ return this.m_labelNames; }

    public HashMap<String, Integer> getLabelNameIndex(){ return this.m_labelNameIndex; }

    public ArrayList<String> getStrings(){ 
    	ArrayList<String> strList = new ArrayList<String>();
    	
    	for(Sequence seq:m_seqList) 
    		strList.add(seq.m_content);
    			
    	return strList; 
    }

    public ArrayList<int[]> getLabels() { 
    	ArrayList<int[]> labelList = new ArrayList<int[]>();
    	
    	for(Sequence seq:m_seqList)
    		labelList.add(seq.getLabels());
    	
    	return labelList; 
    }

    public ArrayList<int[]> getTokens() {
    	ArrayList<int[]> tokenList = new ArrayList<int[]>();
    	
    	for(Sequence seq:m_seqList)
    		tokenList.add(seq.getTokens());
    	
    	return tokenList;
    }

    public ArrayList<Sequence> getSequences() { return this.m_seqList; }

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

    public void loadSequence(String filePath, int maxNum){
    	m_seqList.clear();
    	
        // Read training strings.
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line, token;
            
            while ((line = br.readLine()) != null && line.length() > 0) {
            	Sequence seq = new Sequence(line);
                int idx;
                for(int i = 0 ; i < line.length(); i++){
                	token = Character.toString(line.charAt(i));
                    idx = getTokenIndex(token); //dynamically expand tokenNames: each char to string
                    seq.addToken(token, idx);
                }
                
                m_seqList.add(seq);
                if(m_seqList.size() > maxNum)
                    break;
            }
            System.out.format("[Info]token size: %d\n", m_tokenNames.size());

        } catch (Exception e){
            System.err.format("[Err]File %s doesn't exist.\n", filePath);
        }
    }

    //this is a poor design of loading input files, where we have to assume the alignment by line number
    public void loadLabel(String filePath, int maxNum){
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            String[] labels;
            int lineNo = 0;
            
            while ((line = br.readLine()) != null && line.length() > 0) {
            	Sequence seq = m_seqList.get(lineNo);
            	
                labels = line.split(",");
                int tokenIdx = 0;
                for(String s: labels){
                    if(s.isEmpty())
                        continue;
                    seq.assignLabel(tokenIdx++, getLabelIndex(s));//dynamically expand labelNames
                }
                
                if(++lineNo > maxNum)
                    break;
            }
            getLabelIndex("START");
            getTokenIndex("END");
            System.out.format("[Info]Label size: %d\n", m_labelNames.size());

        } catch (Exception e){
            System.err.format("[Err]File %s doesn't exist.\n", filePath);
        }
    }

    //To create the training sequence
    public String4Learning getStr4Learning(Sequence seq, String mode, Map<Integer, Double> weights){
        // Each string is stored as an object specifying features as table factors.
        String4Learning str;

        HashMap<Integer, ArrayList<Double>> node_features;//x_t and surround token self and isDigit
        HashMap<Integer, ArrayList<Integer>> position_features;//the position of x_t and surround token

        // For each training sample, construct a factor graph, and a list of table factors to specify edge
        // and node features.
        //step 1: construct the graph
        int varNodeSize = seq.getLength();
        int[] label_vec = seq.getLabels();
        Variable[] allVars = new Variable[varNodeSize];
        for(int i = 0; i < allVars.length; i++)
            allVars[i] = new Variable(m_labelNames.size()); //each label variable has this many outcomes. But we can reduce this ahead of time based on the nature of our problem?

        ArrayList<Factor> factorList = new ArrayList<>();   // list of table factors for the current string
        ArrayList<Integer> featureType = new ArrayList<>(); // corresponding feature ID for each list of factors

        //step 2: add node features
        node_features = constructNodeFeature(seq.getTokens());
        ArrayList<Double> cur_feature;
        Factor ptl;
        double[] feature_value_arr = new double[m_labelNames.size()];
        int cur_label, cur_token, feature_idx;
        for(int j = 0; j < varNodeSize; j++) {// for each node/variable

            // note features:
            // token itself:   0 ~ label_size * (token_size + 2) * (num of type%2==0)
            // token is digit: label_size * (token_size + 2) * 5 (=A) ~ A + label_size * (num of type%2==1)
            for (Integer type : node_features.keySet()) {// for each feature type
                //skip feature type without mask
                if(m_mask.containsKey(type))
                    continue;

                cur_feature = node_features.get(type);
                Arrays.fill(feature_value_arr, 1.0);
                cur_token = cur_feature.get(j).intValue();

                if(mode.equals("train")){
                    cur_label = label_vec[j];
                    feature_value_arr[cur_label] = Math.exp(1.0);
                    ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                    factorList.add(ptl);
                    if(type % 2 == 0)// x_t/t-1...: type=0,2,4,6,8
                        featureType.add((m_tokenNames.size() * m_labelNames.size()) * (type / 2) +
                            m_labelNames.size() * cur_token + cur_label);
                    else // is digit: type=1,3,5,7,9
                        featureType.add((m_tokenNames.size() * m_labelNames.size()) * 5 +
                            m_labelNames.size() * 2 * (type / 2) + 2 * cur_token + cur_label);
                } else {//test
                    for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
                        cur_label = label_i;
                        if(type % 2 == 0)
                            feature_idx = (m_tokenNames.size() * m_labelNames.size()) * (type / 2) +
                                m_labelNames.size() * cur_token + cur_label;
                        else
                            feature_idx = (m_tokenNames.size() * m_labelNames.size()) * 5 +
                                    m_labelNames.size() * 2 * (type / 2) + 2 * cur_token + cur_label;
                        if(weights.containsKey(feature_idx))
                            feature_value_arr[cur_label] = Math.exp(weights.get(feature_idx));
                    }
                    ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                    factorList.add(ptl);
                }
            }
        }

        //step 3: add edge features
        int node_feature_size = (m_tokenNames.size() * m_labelNames.size()) * 5 // This is incorrect! As we might mask out some node features!!!!
                + m_labelNames.size() * 2 * 5;
        int curIdx_1, curIdx_2;
        double[] start_feature_arr = new double[m_labelNames.size()];
        double[] trans_feature_arr = new double[m_labelNames.size() * m_labelNames.size()];
        //the size is used to index edge feature such that the index will not overlap for factor graph, we take the largest space for node feature index
        if(!m_mask.containsKey(10)) {
            if(mode.equals("test")){
                Arrays.fill(start_feature_arr, 1.0);
                Arrays.fill(trans_feature_arr, 1.0);
                for (int i = 0; i < m_labelNames.size(); i++) {
                    feature_idx = node_feature_size + 10 + i;
                    if(weights.containsKey(feature_idx))
                        start_feature_arr[i] = Math.exp(weights.get(feature_idx));

                    for(int j = 0; j < m_labelNames.size(); j++) {
                        feature_idx = node_feature_size + 10 + m_labelNames.size()
                                + i * m_labelNames.size() + j;
                        if(weights.containsKey(feature_idx))
                            trans_feature_arr[i * m_labelNames.size() + j] = Math.exp(weights.get(feature_idx));
                    }
                }

                for(int j = 0; j < varNodeSize; j++) {
                    if(j == 0){
                        ptl = LogTableFactor.makeFromValues(
                                new Variable[]{allVars[j]}, start_feature_arr);
                    } else {
                        ptl = LogTableFactor.makeFromValues(
                                new Variable[]{allVars[j-1], allVars[j]}, trans_feature_arr);
                    }
                    factorList.add(ptl);
                }
            } else {
                for(int j = 0; j < varNodeSize; j++) {
                    if(j == 0){
                        curIdx_1 = getLabelIndex("START");
                        curIdx_2 = label_vec[j];
                        Arrays.fill(start_feature_arr, 1.0);
                        start_feature_arr[curIdx_2] = Math.exp(1.0);
                        ptl = LogTableFactor.makeFromValues(
                                new Variable[]{allVars[j]}, start_feature_arr);
                        factorList.add(ptl);
                        featureType.add(node_feature_size + 10 + curIdx_2);
                    } else {
                        curIdx_1 = label_vec[j - 1];
                        curIdx_2 = label_vec[j];
                        trans_feature_arr = label_transition(curIdx_1, curIdx_2);
                        ptl = LogTableFactor.makeFromValues(
                                new Variable[]{allVars[j-1], allVars[j]}, trans_feature_arr);
                        factorList.add(ptl);
                        featureType.add(node_feature_size + 10 + m_labelNames.size()
                                + curIdx_1 * m_labelNames.size() + curIdx_2);
                    }
                }
            }
        }

        //step 3: add position features
        int cur_feature_size = node_feature_size + 10 + m_labelNames.size()
                + m_labelNames.size() * m_labelNames.size();
        position_features = constructPositionFeature(seq.getTokens());
        ArrayList<Integer> pos_feature;
        for(int j = 0; j < varNodeSize; j++) {// for each node/variable
            for (Integer type : position_features.keySet()) {// for each feature type
                //skip feature type with mask
                if(m_mask.containsKey(type))//11-22
                    continue;

                pos_feature = position_features.get(type);

                if (mode.equals("train")) {
                    cur_token = pos_feature.get(j);
                    cur_label = label_vec[j];
                    Arrays.fill(feature_value_arr, 1.0);
                    feature_value_arr[cur_label] = Math.exp(1.0);
                    ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                    factorList.add(ptl);
                    featureType.add(cur_feature_size + 10 + m_labelNames.size() * 2 * (type-11)
                             + 2 * cur_token + cur_label);

                } else {
//                    for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
//                        Arrays.fill(feature_value_arr, 1.0);
//                        feature_value_arr[label_i] = Math.exp(pos_feature.get(j));
//                        ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
//                        factorList.add(ptl);
//                        featureType.add(cur_feature_size + 10 + m_labelNames.size() * (type-11) + label_i);
//                    }
                    Arrays.fill(feature_value_arr, 1.0);
                    cur_token = pos_feature.get(j);
                    for(int label_i = 0; label_i < m_labelNames.size(); label_i++) {
                        cur_label = label_i;
                        feature_idx = cur_feature_size + 10 + m_labelNames.size() * 2 * (type-11)
                                + 2 * cur_token + cur_label;
                        if(weights.containsKey(feature_idx))
                            feature_value_arr[cur_label] = Math.exp(weights.get(feature_idx));
                    }
                    ptl = LogTableFactor.makeFromValues(new Variable[]{allVars[j]}, feature_value_arr);
                    factorList.add(ptl);
                }
            }
        }

        str = new String4Learning(factorList, featureType, label_vec);

        return str;
    }

    // build the node features
    public HashMap<Integer, ArrayList<Double>> constructNodeFeature(int[] tokens){
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
        for(int i = 0;i < tokens.length; i++){
            //x_t
            curIdx = tokens[i];
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
                curIdx = tokens[i-1];
            curToken = m_tokenNames.get(curIdx);
            x_t_pre_1.add((double) curIdx);
            //x_t-1 is digit
            if(isDigit(curToken))
                x_t_pre_1_is_digit.add(1.0);
            else
                x_t_pre_1_is_digit.add(0.0);

            //x_t-2
            if(i < 2)
                curIdx = getTokenIndex("START");
            else
                curIdx = tokens[i-2];
            curToken = m_tokenNames.get(curIdx);
            x_t_pre_2.add((double) curIdx);
            //x_t-2 is digit
            if(isDigit(curToken))
                x_t_pre_2_is_digit.add(1.0);
            else
                x_t_pre_2_is_digit.add(0.0);

            //x_t+1
            if(i == tokens.length-1)
                curIdx = getTokenIndex("END");
            else
                curIdx = tokens[i+1];
            curToken = m_tokenNames.get(curIdx);
            x_t_next_1.add((double) curIdx);
            //x_t+1 is digit
            if(isDigit(curToken))
                x_t_next_1_is_digit.add(1.0);
            else
                x_t_next_1_is_digit.add(0.0);

            //x_t+2
            if(i == tokens.length-1 || i == tokens.length-2)
                curIdx = getTokenIndex("END");
            else
                curIdx = tokens[i+2];
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

    // build the position features: denoting which part each token belongs (Zheng's)
    public HashMap<Integer, ArrayList<Integer>> constructPositionFeature(int[] tokens){

        int num_part = 4;
        int[] position_idxs = new int[tokens.length];
        for(int i = 0; i < tokens.length; i++){
            position_idxs[i] = i / num_part;
        }

        HashMap<Integer, ArrayList<Integer>> position_features = new HashMap<>();

        ArrayList<Integer> x_t_part0 = new ArrayList<>();
        ArrayList<Integer> x_t_part1 = new ArrayList<>();
        ArrayList<Integer> x_t_part2 = new ArrayList<>();
        ArrayList<Integer> x_t_part3 = new ArrayList<>();

        ArrayList<Integer> x_t_pre_1_part0 = new ArrayList<>();
        ArrayList<Integer> x_t_pre_1_part1 = new ArrayList<>();
        ArrayList<Integer> x_t_pre_1_part2 = new ArrayList<>();
        ArrayList<Integer> x_t_pre_1_part3 = new ArrayList<>();

        ArrayList<Integer> x_t_next_1_part0 = new ArrayList<>();
        ArrayList<Integer> x_t_next_1_part1 = new ArrayList<>();
        ArrayList<Integer> x_t_next_1_part2 = new ArrayList<>();
        ArrayList<Integer> x_t_next_1_part3 = new ArrayList<>();

        int curIdx;
        String curToken;
        for(int i = 0;i < tokens.length; i++){
            if(position_idxs[i] == 0)
                x_t_part0.add(1);
            else
                x_t_part0.add(0);

            if(position_idxs[i] == 1)
                x_t_part1.add(1);
            else
                x_t_part1.add(0);

            if(position_idxs[i] == 2)
                x_t_part2.add(1);
            else
                x_t_part2.add(0);

            if(position_idxs[i] == 3)
                x_t_part3.add(1);
            else
                x_t_part3.add(0);

            //x_t-1
            if(i == 0) {
                x_t_pre_1_part0.add(1);
                x_t_pre_1_part1.add(0);
                x_t_pre_1_part2.add(0);
                x_t_pre_1_part3.add(0);
            } else {
                if(position_idxs[i-1] == 0)
                    x_t_pre_1_part0.add(1);
                else
                    x_t_pre_1_part0.add(0);

                if(position_idxs[i-1] == 1)
                    x_t_pre_1_part1.add(1);
                else
                    x_t_pre_1_part1.add(0);

                if(position_idxs[i-1] == 2)
                    x_t_pre_1_part2.add(1);
                else
                    x_t_pre_1_part2.add(0);

                if(position_idxs[i-1] == 3)
                    x_t_pre_1_part3.add(1);
                else
                    x_t_pre_1_part3.add(0);
            }

            //x_t+1
            if(i == tokens.length-1) {
                x_t_next_1_part0.add(0);
                x_t_next_1_part1.add(0);
                x_t_next_1_part2.add(0);
                x_t_next_1_part3.add(1);
            } else {
                if(position_idxs[i+1] == 0)
                    x_t_next_1_part0.add(1);
                else
                    x_t_next_1_part0.add(0);

                if(position_idxs[i+1] == 1)
                    x_t_next_1_part1.add(1);
                else
                    x_t_next_1_part1.add(0);

                if(position_idxs[i+1] == 2)
                    x_t_next_1_part2.add(1);
                else
                    x_t_next_1_part2.add(0);

                if(position_idxs[i+1] == 3)
                    x_t_next_1_part3.add(1);
                else
                    x_t_next_1_part3.add(0);
            }
        }

        position_features.put(11,x_t_part0);
        position_features.put(12, x_t_part1);
        position_features.put(13, x_t_part2);
        position_features.put(14, x_t_part3);

        position_features.put(15, x_t_pre_1_part0);
        position_features.put(16, x_t_pre_1_part1);
        position_features.put(17, x_t_pre_1_part2);
        position_features.put(18, x_t_pre_1_part3);

        position_features.put(19, x_t_next_1_part0);
        position_features.put(20, x_t_next_1_part1);
        position_features.put(21, x_t_next_1_part2);
        position_features.put(22, x_t_next_1_part3);

        return position_features;
    }

    // This is the first-order edge feature enumerating all possible label transitions.
    public double[] label_transition(int label1, int label2){
        int len_arr = m_labelNames.size() * m_labelNames.size();
        double[] arr = new double[len_arr];
        Arrays.fill(arr, 1.0);
        arr[label1*m_labelNames.size()+label2] = Math.exp(1.0);
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

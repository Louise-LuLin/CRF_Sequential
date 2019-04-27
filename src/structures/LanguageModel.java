/**
 * 
 */
package structures;

import java.util.HashMap;
import java.util.Map;

/**
 * @author hongning
 * Suggested structure for constructing N-gram language model
 */
public class LanguageModel {

	int m_N; // N-gram
	int m_V; // the vocabular size
    int m_D; // the doc size
    int m_Min; // min count
    int m_Max; // max count of seen Ngram
	HashMap<String, Token> m_model; // sparse structure for storing the maximum likelihood estimation of LM with the seen N-grams
	HashMap<String, Double> m_S; // seen word types occur after w_{i-1}...w_{i-n+1}
    LanguageModel m_reference; // pointer to the reference language model for smoothing purpose
	
	double m_lambda; // parameter for linear interpolation smoothing
	double m_delta; // parameter for absolute discount smoothing

    int[] m_D4Class;
    int m_classNum;

	public LanguageModel(int N, double lambda, double delta) {
		m_N = N;
        m_lambda = lambda;
        m_delta = delta;
		m_model = new HashMap<String, Token>();
	}

    public void setReference(LanguageModel ref){
        this.m_reference = ref;
    }

    public LanguageModel getReference(){
        return this.m_reference;
    }

	public void setModel(HashMap<String, Token> NStats){
        this.m_S = new HashMap<>();
        int totalCount = 0;
        int maxCount = 0;
        int minCount = 100;
        for (Map.Entry<String, Token> entry : NStats.entrySet()) {
            int curCount = (int) entry.getValue().getValue();
            totalCount += curCount;
            if(maxCount < curCount){
                maxCount = curCount;
            }
            if(minCount > curCount){
                minCount = curCount;
            }
            if(m_N > 1) {
                String token = entry.getKey();
                String pre = token.substring(0, token.lastIndexOf('-'));
                if (this.m_S.containsKey(pre)) {
                    this.m_S.put(pre, this.m_S.get(pre) + 1.0);
                } else {
                    this.m_S.put(pre, 1.0);
                }
            }
        }

        this.m_Max = maxCount;
        this.m_Min = minCount;
        this.m_D = totalCount;
		this.m_model = new HashMap<>(NStats);
		this.m_V = NStats.size();
    }

    public void setModel(HashMap<String, Token> NStats, int[] classStat, int V){
	    this.m_model = new HashMap<>(NStats);
	    this.m_D4Class = classStat.clone();
	    this.m_V = V;
    }

    public HashMap<String, Token>  getModel(){
        return this.m_model;
    }
	
	public double calcMLProb(String token) {
        if(m_N > 1){
            String pre = token.substring(0, token.lastIndexOf('-'));
            if(m_model.containsKey(token)) {
                return m_model.get(token).getValue() / m_reference.getModel().get(pre).getValue();
            } else{
                return 0.0;
            }

        } else {
            if(m_model.containsKey(token)) {
                return m_model.get(token).getValue() / m_D;
            } else{
                return 0.0;
            }
        }
	}

    public double calcAdditiveSmoothedProb(String token){
        if(m_N > 1){
            String pre = token.substring(0, token.lastIndexOf("-"));
            if(m_model.containsKey(token)){
                return (m_model.get(token).getValue() + m_delta) /
                        (m_reference.getModel().get(pre).getValue()
                        + m_delta * m_V);
            } else{
                if(m_reference.getModel().containsKey(pre)) {
                    return m_delta / (m_reference.getModel().get(pre).getValue()
                            + m_delta * m_V);
                }else{
                    return 1.0/ m_V;
                }
            }

        } else{
            // additive smoothing to smooth a unigram language model
            if(m_model.containsKey(token)) {
                return (m_model.get(token).getValue() + m_delta) /
                        (m_D + m_delta * m_V);
            } else{
                return m_delta / (m_D + m_delta * m_V);
            }
        }
    }

	public double calcLinearSmoothedProb(String token) {
		if (m_N > 1) {
            String N_1gram = token.substring(token.indexOf('-') + 1, token.length());
            return (1.0 - m_lambda) * calcMLProb(token)
                    + m_lambda * m_reference.calcLinearSmoothedProb(N_1gram);
        }
		else {
            // additive smoothing to smooth a unigram language model
            if(m_model.containsKey(token)) {
                return (m_model.get(token).getValue() + m_delta) / (m_D + m_delta * m_V);
            } else{
                return m_delta / (m_D + m_delta * m_V);
            }
        }
	}

    public double calcAbsoluteSmoothedProb(String token){
        if (m_N > 1) {
            String pre = token.substring(0, token.lastIndexOf('-'));
            String N_1gram = token.substring(token.indexOf('-') + 1, token.length());

            if(m_reference.getModel().containsKey(pre)) {
                double count = m_reference.getModel().get(pre).getValue();
                double lambda = 0;
                if(m_S.containsKey(pre)) {
                    lambda = m_delta * m_S.get(pre) / count;
                } else{
                    lambda = 1.0;
                }
                if(m_model.containsKey(token)) {
                    return (m_model.get(token).getValue() - m_delta) / count
                            + lambda * m_reference.calcAbsoluteSmoothedProb(N_1gram);
                } else {
                    return lambda * m_reference.calcAbsoluteSmoothedProb(N_1gram);
                }
            } else{
                // if N_1 gram is not seen, than back off to N_1gram model
                return m_reference.calcAbsoluteSmoothedProb(N_1gram);
            }
        }
        else {
            // additive smoothing to smooth a unigram language model
            if(m_model.containsKey(token)) {
                return (m_model.get(token).getValue() + m_delta) / (m_D + m_delta * m_V);
            } else{
                return  m_delta / (m_D + m_delta * m_V);
            }
        }
    }
}

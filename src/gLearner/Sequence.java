package gLearner;

import java.util.ArrayList;
import structures.Token;

public class Sequence {
	String m_content;
	
	ArrayList<Token> m_tokenList;
	
	public Sequence(String content) {
		m_content = content;
		
		m_tokenList = new ArrayList<Token>();
	}

	void setTokenList(ArrayList<Token> tokenlist){
		this.m_tokenList = tokenlist;
	}

	void setLabelIDs(int[] labelIDs){
		for(int i = 0; i < m_tokenList.size(); i++){
			m_tokenList.get(i).setLabelID(labelIDs[i]);
		}
	}

	void setLabels(String[] labels){
		ArrayList<Token> tokenlist = new ArrayList<>();
		for(int i = 0; i < m_tokenList.size(); i++){
			m_tokenList.get(i).setLabel(labels[i]);
		}
	}

	String getContent(){ return m_content; }
	
	void addToken(String content, int index) {
		m_tokenList.add(new Token(content, index));
	}
	
	void assignLabel(int tokenIdx, String label, int labelID) {
		if (tokenIdx < m_tokenList.size()) {
			m_tokenList.get(tokenIdx).setLabel(label);
			m_tokenList.get(tokenIdx).setLabelID(labelID);
		}
		else 
			System.err.format("Token index %d out of sequence range %d!", tokenIdx, m_tokenList.size());
	}
	
	int[] getLabelIDs() {
		int[] labels = new int[m_tokenList.size()];
		
		for(int i=0; i<labels.length; i++)
			labels[i] = m_tokenList.get(i).getLabelID();
		
		return labels;
	}

	public String[] getLabels() {
		String[] labels = new String[m_tokenList.size()];

		for(int i=0; i<labels.length; i++)
			labels[i] = m_tokenList.get(i).getLabel();

		return labels;
	}
	
	int[] getTokenIDs() {
		int[] tokens = new int[m_tokenList.size()];
		
		for(int i=0; i<tokens.length; i++)
			tokens[i] = m_tokenList.get(i).getIndex();
		
		return tokens;
	}

	public String[] getTokens() {
		String[] tokens = new String[m_tokenList.size()];

		for(int i=0; i<tokens.length; i++)
			tokens[i] = m_tokenList.get(i).getToken();

		return tokens;
	}

	int[] getLabelIDs(int min, int max) {
		int[] labels = new int[max-min];

		for(int i=min; i<max; i++)
			labels[i] = m_tokenList.get(i).getLabelID();

		return labels;
	}

	int[] getTokens(int min, int max) {
		int[] tokens = new int[max-min];

		for(int i=min; i<max; i++)
			tokens[i] = m_tokenList.get(i).getIndex();

		return tokens;
	}

	public Sequence getSubSeq(int min, int max){
		String content = this.getContent().substring(min, max);
		Sequence sub = new Sequence(content);
		sub.setTokenList(new ArrayList<Token>(this.m_tokenList.subList(min, max)));
		return sub;
	}

	int getLength() {
		return m_tokenList.size();
	}
}

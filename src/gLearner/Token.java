package gLearner;

public class Token {
	String m_content;//text content of the token
	int m_index;//index of the content word in the whole vocabulary	
	int m_label;//index of the label in the whole label set
	
	public Token(String content, int index, int label) {
		m_content = content;
		m_index = index;
		m_label = label;
	}
	
	public Token(String content, int index) {
		m_content = content;
		m_index = index;
		m_label = -1;
	}

	public void setLabel(int label){
		m_label = label;
	}
}

/**
 * 
 */
package structures;

public class Token {
	String m_token;//text content of the token
	int m_id;//index of the content word in the whole vocabulary
	String m_label; // label context of the token
	int m_label_id;//index of the label in the whole label set
	double m_value; // frequency or count of this token/N-gram

	//default constructor
	public Token(String token) {
		m_token = token;
		m_id = -1;
		m_value = 0;
	}

	public Token(String content, int index, int labelid) {
		m_token = content;
		m_id = index;
		m_label_id = labelid;
	}

	public Token(String content, int index) {
		m_token = content;
		m_id = index;
		m_label = "";
		m_label_id = -1;
	}

	public String getToken() {
		return m_token;
	}

	public void setToken(String token) {
		this.m_token = token;
	}

	public double getValue() {
		return m_value;
	}

	public void setValue(double value) {
		this.m_value =value;
	}

	public void addTF(int value){
		this.m_value += value;
	}

	public void setLabelID(int label_id){
		m_label_id = label_id;
	}

	public void setLabel(String label){
		m_label = label;
	}

	public String getLabel(){ return m_label; }

	public int getLabelID() { return m_label_id; }

	public int getIndex() { return m_id; }

	public void setIndex(int index){ this.m_id = index; }
}

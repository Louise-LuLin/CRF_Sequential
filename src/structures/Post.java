/**
 * 
 */
package structures;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * @author hongning
 * @version 0.1
 * @category data structure
 * data structure for a Yelp review document
 * You can create some necessary data structure here to store the processed text content, e.g., bag-of-word representation
 */
public class Post {
	//unique review ID from Yelp
	String m_ID;		
	public void setID(String ID) {
		m_ID = ID;
	}
	
	public String getID() {
		return m_ID;
	}

	//author's displayed name
	String m_author;	
	public String getAuthor() {
		return m_author;
	}

	public void setAuthor(String author) {
		this.m_author = author;
	}
	
	//author's location
	String m_location;
	public String getLocation() {
		return m_location;
	}

	public void setLocation(String location) {
		this.m_location = location;
	}

	//review text content
	String m_content;
	public String getContent() {
		return m_content;
	}

	public void setContent(String content) {
		if (!content.isEmpty())
			this.m_content = content;
	}
	
	public boolean isEmpty() {
		return m_content==null || m_content.isEmpty();
	}

	//timestamp of the post
	String m_date;
	public String getDate() {
		return m_date;
	}

	public void setDate(String date) {
		this.m_date = date;
	}
	
	//overall rating to the business in this review
	double m_rating;
	public double getRating() {
		return m_rating;
	}

	public void setRating(double rating) {
		this.m_rating = rating;
	}

	public Post(String ID) {
		m_ID = ID;
	}
	
	String[] m_tokens; // we will store the tokens 
	public String[] getTokens() {
		return m_tokens;
	}
	public void setTokens(String[] tokens) {
		m_tokens = tokens;
	}

	String[] m_POSs;
	public String[] getPOSs(){
		return m_POSs;
	}
	public void setPOSs(String[] poss){ m_POSs = poss;}
	
	HashMap<String, Token> m_vector; // suggested sparse structure for storing the vector space representation with N-grams for this document
	public HashMap<String, Token> getVct() {
		return m_vector;
	}
	
	public void setVct(HashMap<String, Token> vct) {
		m_vector = vct;
	}

	int[] m_hashCode;
	public void setHash(int[] hash){ m_hashCode = hash; }
	public int[] getHash(){ return m_hashCode; }
	
	public double similiarity(Post p) {
        Set<String> intersect = new HashSet<>(this.getVct().keySet());
        intersect.retainAll(p.getVct().keySet());

        double dotProduct = 0.0;
        for(String key : intersect){
            dotProduct += this.getVct().get(key).getValue() * p.getVct().get(key).getValue();
        }

        double dotp1 = 0.0;
        for(Token tk : this.getVct().values()){
            dotp1 += Math.pow(tk.getValue(), 2);
        }

        double dotp2 = 0.0;
        for(Token tk : p.getVct().values()){
            dotp2 += Math.pow(tk.getValue(), 2);
        }

        double cosine = 0.0;
        if (dotp1 <= 0.0 || dotp2 <= 0.0){
            cosine = 0.0;
        } else {
            cosine = dotProduct / (Math.sqrt(dotp1) * Math.sqrt(dotp2));
        }

		return cosine;//compute the cosine similarity between this post and input p based on their vector space representation
	}

	public double similiarityHash(Post p) {
		double dotProduct = 0.0;
		for(int i = 0; i < m_hashCode.length; i++){
			dotProduct += this.getHash()[i] * p.getHash()[i];
		}

		double dotp1 = 0.0;
		for(int i = 0; i < m_hashCode.length; i++){
			dotp1 += Math.pow(m_hashCode[i], 2);
		}

		double dotp2 = 0.0;
		for(int i = 0; i < p.getHash().length; i++){
			dotp2 += Math.pow(p.getHash()[i], 2);
		}

		double cosine = 0.0;
		if (dotp1 <= 0.0 || dotp2 <= 0.0){
			cosine = 0.0;
		} else {
			cosine = dotProduct / (Math.sqrt(dotp1) * Math.sqrt(dotp2));
		}

		return cosine;//compute the cosine similarity between this post and input p based on their vector space representation
	}

}

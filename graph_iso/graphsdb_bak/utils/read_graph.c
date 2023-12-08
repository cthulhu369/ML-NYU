/*
 The graph file format
 The graphs are stored in a compact binary format, one graph per file. The file is composed of 16 bit words, 
 which are represented using the so-called little-endian convention, i.e. the least significant byte of the word is stored first.
 
 Two different formats are used for labeled and unlabeled graphs. 
 In labeled graphs, we assume that attributes are represented by integer numbers. Also, the use of floating point values can be 
 somewhat more problematic, because:
 -	usually it makes little sense to compare floating point numbers for strict equality 
 -	there is no universally useable binary format for storing binary numbers (there are still machines or programming languages 
	that do not support IEEE 754 out there,	and converting from a floating point format to another can be quite a nightmare); 
	hence we should stick to a text format, which requires quite more space.
 
 These inconvenients are not repaid for by significant advantages, since integer attributes can also be used to perform arithmetic 
 tasks, e.g. for a distance evaluation. The most important parameter characterizing the difficulty of the matching problem is the number 
 N of different attribute values: obviously the higher this number, the easier is the matching problem. It is important to have in 
 the database different values of N.
 
 In order to avoid the need to have several copies of the database with different values of N, we can exploit the following idea: 
 each (node or edge) attribute is generated as a 16 bit value, Then, we can make experiments with any N of the form 2^k, for k not 
 greater than 16, just by using, in the attribute comparison function, only the first k bits of the attribute.
 
 With this technique we made the attributed graph database that is general enough for experimenting with many different attribute 
 cardinalities, without incurring in the explosion of the size required to store the database.
 
 Following is the labeled graph file format.
 The first word contains the number of nodes in the graph; this means that this format can deal with graph of up to 65535 nodes 
 (however, actual graphs in the database are quite smaller, up to 100 nodes).
 Then follow N words, that are the attribute of corresponding node. Then, for each node i, the file contains the number of edges coming 
 out of the node itself (Ei) followed by 2*Ei words that code the edges. In particular, for each edge, the first word represents the 
 destination node of the edge and the second word represents the edge attribute. Node numeration is 0-based, so the first node of 
 the graph has index 0.
 
 The following C code shows how a graph file can be read into an adjacency matrix, a vector representing the nodes attribute and a 
 matrix representing the edges attribute; the code assumes that the input file has been opened using the binary mode
 
 */


/* WARNING: for simplicity this code does not check for End-Of-File! */
#include <stdio.h>

#define MAX 1024  // Adjust this value based on your needs

unsigned short read_word(FILE *in)
{ unsigned char b1, b2;
	
    b1=getc(in); /* Least-significant Byte */
    b2=getc(in); /* Most-significant Byte */
	
    return b1 | (b2 << 8);
}

/* This function assumes that the adjacency matrix is statically allocated.
 * The return value is the number of nodes in the graph.
 */
int read_graph(FILE *in, int matrix[MAX][MAX], int node_attr[MAX], int edge_attr[MAX][MAX])
{
    int nodes;
    int edges;
    int target;
	
    /* Read the number of nodes */
    nodes = read_word(in);
	
    /* Read the attributes of nodes */
    for(int i=0; i<nodes; i++)
	{ node_attr[i] = read_word(in);
	}
	
    /* Clean-up the matrix */
    for(int i=0; i<nodes; i++)
		for(int j=0; j<nodes; j++)
			matrix[i][j]=0;
	
    /* For each node i... */
    for(int i=0; i<nodes; i++)
	{
		/* Read the number of edges coming out of node i */
		edges=read_word(in);
		
		/* For each edge out of node i... */
		for(int j=0; j<edges; j++)
		{
			/* Read the destination node of the edge */
			target = read_word(in);
			
			/* Insert the edge in the adjacency matrix */
			matrix[i][target] = 1;
			
			/* Read the edge attribute and insert in the matrix that holds the attributes of edges */
			edge_attr[i][target] = read_word(in);
		}
	}
	
	
    return nodes;
}

int main() {
    FILE *file;
    int matrix[MAX][MAX];
    int node_attr[MAX];
    int edge_attr[MAX][MAX];

    // Open the graph file
    file = fopen("/home/ericp/Downloads/graphsdb/iso/m2D/m2D/iso_m2D_m1024.A00", "rb");  // Replace with your actual file name
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Read the graph into the matrix
    int nodes = read_graph(file, matrix, node_attr, edge_attr);

    // Do something with the graph data...
    // For example, print the number of nodes
    printf("Number of nodes: %d\n", nodes);

    // Close the file
    fclose(file);

    return 0;
}

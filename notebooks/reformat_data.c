//gcc -lgsl reformat_data.c

/* c headers */
#include <stdio.h>
#include <math.h>

/* gsl headers */
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

void print_vector(gsl_vector *v);
void print_matrix(gsl_matrix *m, int rows, int cols);

int main(void) {
    //Find out format of Puerrer's data
    double mij;
    int nx = 133;
    FILE *f = fopen("/home/lppekows/projects/aligo/lal-data/lalsimulation/SEOBNRv2ROM_DS_sub1_Bamp_bin.dat", "rb");
    gsl_matrix *m = gsl_matrix_alloc(nx, nx);
    gsl_matrix_fread(f, m);
    //print_matrix(m, 10, 133);
    print_matrix(m, 1, 1);
    fclose(f);
    
    //Read in amplitude matrix as text file and save it as a gsl_matrix binary file
    int namp = 12;
    int nt = 73624;
    FILE *f_amp1 = fopen("Bamp_matrix.txt", "r");
    gsl_matrix *m_amp = gsl_matrix_alloc(namp, nt);
    gsl_matrix_fscanf(f_amp1, m_amp);

    FILE *f_amp2 = fopen("Bamp_matrix.dat", "wb");
    gsl_matrix_fwrite(f_amp2, m_amp);
    fclose(f_amp2);

    FILE *f_amp3 = fopen("Bamp_matrix.dat", "rb");
    gsl_matrix *m_amp3 = gsl_matrix_alloc(namp, nt);
    gsl_matrix_fread(f_amp3, m_amp3);
    //print_matrix(m_amp3, 1, nt);
    print_matrix(m_amp3, 1, 5);

    //Read in phase matrix as text file and save it as a gsl_matrix binary file
    int nphase = 7;
    FILE *f_phase1 = fopen("Bphase_matrix.txt", "r");
    gsl_matrix *m_phase = gsl_matrix_alloc(nphase, nt);
    gsl_matrix_fscanf(f_phase1, m_phase);

    FILE *f_phase2 = fopen("Bphase_matrix.dat", "wb");
    gsl_matrix_fwrite(f_phase2, m_phase);
    fclose(f_phase2);

    FILE *f_phase3 = fopen("Bphase_matrix.dat", "rb");
    gsl_matrix *m_phase3 = gsl_matrix_alloc(nphase, nt);
    gsl_matrix_fread(f_phase3, m_phase3);
    //print_matrix(m_phase3, 1, nt);
    print_matrix(m_phase3, 1, 5);
    
    //Read in times and save it as gsl_vector    
    FILE *f_t1 = fopen("times.txt", "r");
    gsl_vector *v_t1 = gsl_vector_alloc(nt);
    gsl_vector_fscanf(f_t1, v_t1);

    FILE *f_t2 = fopen("times.dat", "wb");
    gsl_vector_fwrite(f_t2, v_t1);
    fclose(f_t2);

    FILE *f_t3 = fopen("times.dat", "rb");
    gsl_vector *v_t3 = gsl_vector_alloc(nt);
    gsl_vector_fread(f_t3, v_t3);
    print_vector(v_t3);

    //Read in amplitude coefficients and save it as gsl_vector
    int neta = 16;  
    int nlam1 = 16;
    int nlam2 = 16;
    FILE *f_ciamp1 = fopen("Amp_ciall.txt", "r");
    gsl_vector *v_ciamp1 = gsl_vector_alloc(namp*neta*nlam1*nlam2);
    gsl_vector_fscanf(f_ciamp1, v_ciamp1);

    FILE *f_ciamp2 = fopen("Amp_ciall.dat", "wb");
    gsl_vector_fwrite(f_ciamp2, v_ciamp1);
    fclose(f_ciamp2);

    FILE *f_ciamp3 = fopen("Amp_ciall.dat", "rb");
    gsl_vector *v_ciamp3 = gsl_vector_alloc(namp*neta*nlam1*nlam2);
    gsl_vector_fread(f_ciamp3, v_ciamp3);
    print_vector(v_ciamp3);
    
    //Read in phase coefficients and save it as gsl_vector 
    FILE *f_ciphase1 = fopen("Phase_ciall.txt", "r");
    gsl_vector *v_ciphase1 = gsl_vector_alloc(nphase*neta*nlam1*nlam2);
    gsl_vector_fscanf(f_ciphase1, v_ciphase1);

    FILE *f_ciphase2 = fopen("Phase_ciall.dat", "wb");
    gsl_vector_fwrite(f_ciphase2, v_ciphase1);
    fclose(f_ciphase2);

    FILE *f_ciphase3 = fopen("Phase_ciall.dat", "rb");
    gsl_vector *v_ciphase3 = gsl_vector_alloc(nphase*neta*nlam1*nlam2);
    gsl_vector_fread(f_ciphase3, v_ciphase3);
    print_vector(v_ciphase3);
 
    return 0;
}

/*******************/
/* Print a vector. */
/*******************/
void print_vector(gsl_vector *v)
{
  int n;
  int i;

  n = (*v).size;
  
  for (i = 0; i < n; i++){
    printf ("%8g  ", gsl_vector_get (v, i));
  } 
  printf("\n\n");
}


/*******************/
/* Print a matrix. */
/*******************/
void print_matrix(gsl_matrix *m, int rows, int columns)
{
  //int rows, columns;
  int i, j;

  //rows = (*m).size1;
  //columns = (*m).size2;
  
  printf("[\n");
  for (i = 0; i < rows; i++){
    printf("[");
    for (j = 0; j < columns; j++) 
      //printf ("%8g  ", gsl_matrix_get (m, i, j));
      printf ("%.18e  ", gsl_matrix_get (m, i, j));
    printf ("]\n\n");
  }
  printf("]\n\n");
}


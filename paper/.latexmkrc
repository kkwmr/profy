$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error';

# Compile in-place (no separate build directory)
$out_dir = '.';
$aux_dir = '.';

# Extra files to clean on `latexmk -c`
$clean_ext .= ' %R.run.xml %R.synctex.gz';


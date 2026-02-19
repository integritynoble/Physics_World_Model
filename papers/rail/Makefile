.PHONY: pdf clean

pdf:
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof sections/*.aux

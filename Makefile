all:
	latexmk -lualatex -latexoption="-synctex=1" -silent main.tex 
v:
	latexmk -lualatex -latexoption="-synctex=1" main.tex

clean:
	latexmk -c

purge:
	latexmk -C

count:
	@echo -n "TODO: "
	@find . -name "*.tex" | xargs grep -n -F "\todo" |  py -x "x if r'\todo' in x.split('%', 1)[0] else None" | wc -l
	@echo -n "FIGS: "
	@find . -name "*.tex" | xargs grep -F "\missingfigure" | py -x "x.split('%', 1)[-1]" | grep -F "\missingfigure" | wc -l

check:
	@find . -name "*tex" | xargs -I [] grep \" [] | grep -v "\`\`" && exit 1 || exit 0

txt:
	pandoc -f latex --bibliography=references.bib -t plain -o thesis.txt main.tex

odt:
	pandoc -f latex --bibliography=references.bib -t odt -o thesis.odt main.tex
force:
	latexmk -f -lualatex -latexoption="-synctex=1" main.tex


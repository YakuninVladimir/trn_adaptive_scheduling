% Clean-room compatible class for the MSU thesis template.
% It preserves the user-facing macro names used by the yes-science/msu-thesis template,
% but is intentionally simplified for maintainability and Overleaf/local builds.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{BYUPhys}[2026/04/27 Clean MSU thesis class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper,12pt]{article}

\RequirePackage[a4paper,left=30mm,right=15mm,top=20mm,bottom=20mm]{geometry}
\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{ifthen}
\RequirePackage{tocbibind}
\RequirePackage[unicode,hidelinks]{hyperref}
\RequirePackage{bookmark}

\makeatletter

% Metadata fields compatible with the source template.
\newcommand{\University}[1]{\gdef\@University{#1}}
\newcommand{\Faculty}[1]{\gdef\@Faculty{#1}}
\newcommand{\Chair}[1]{\gdef\@Chair{#1}}
\newcommand{\Lab}[1]{\gdef\@Lab{#1}}
\newcommand{\GrText}[1]{\gdef\@GrText{#1}}
\newcommand{\AcadGroup}[1]{\gdef\@AcadGroup{#1}}
\newcommand{\AdvisorText}[1]{\gdef\@AdvisorText{#1}}
\newcommand{\ConsultantText}[1]{\gdef\@ConsultantText{#1}}
\newcommand{\Year}[1]{\gdef\@Year{#1}}
\newcommand{\Month}[1]{\gdef\@Month{#1}}
\newcommand{\City}[1]{\gdef\@City{#1}}
\newcommand{\degree}[1]{\gdef\@degree{#1}}
\newcommand{\docname}[1]{\gdef\@docname{#1}}
\newcommand{\TitleTop}[1]{\gdef\@TitleTop{#1}}
\newcommand{\TitleMiddle}[1]{\gdef\@TitleMiddle{#1}}
\newcommand{\TitleBottom}[1]{\gdef\@TitleBottom{#1}}
\newcommand{\TitleTopEng}[1]{\gdef\@TitleTopEng{#1}}
\newcommand{\TitleMiddleEng}[1]{\gdef\@TitleMiddleEng{#1}}
\newcommand{\TitleBottomEng}[1]{\gdef\@TitleBottomEng{#1}}
\newcommand{\TitlePageText}[1]{\gdef\@TitlePageText{#1}}
\newcommand{\Abstract}[1]{\gdef\@Abstract{#1}}
\newcommand{\AbstractEng}[1]{\gdef\@AbstractEng{#1}}
\newcommand{\AbstractText}[1]{\gdef\@AbstractText{#1}}
\newcommand{\Acknowledgments}[1]{\gdef\@Acknowledgments{#1}}
\newcommand{\AcknowledgmentsText}[1]{\gdef\@AcknowledgmentsText{#1}}
\newcommand{\AuthorText}[1]{\gdef\@AuthorText{#1}}
\newcommand{\Author}[1]{\gdef\@Author{#1}}
\newcommand{\AuthorEng}[1]{\gdef\@AuthorEng{#1}}
\newcommand{\AuthorGenitiveCase}[1]{\gdef\@AuthorGenitiveCase{#1}}
\newcommand{\Advisor}[1]{\gdef\@Advisor{#1}}
\newcommand{\AdvisorDegree}[1]{\gdef\@AdvisorDegree{#1}}
\newcommand{\Consultant}[1]{\gdef\@Consultant{#1}}
\newcommand{\ConsultantDegree}[1]{\gdef\@ConsultantDegree{#1}}
\newcommand{\ListingText}[1]{\gdef\@ListingText{#1}}
\newcommand{\AlgorithmText}[1]{\gdef\@AlgorithmText{#1}}

% Defaults.
\University{Московский государственный университет имени М.В. Ломоносова}
\Faculty{Факультет}
\Chair{Кафедра}
\GrText{гр.}
\AcadGroup{}
\AdvisorText{Научный руководитель}
\ConsultantText{Научный консультант}
\Year{\the\year}
\Month{}
\City{Москва}
\docname{Выпускная квалификационная работа}
\TitleTop{Название работы}
\TitlePageText{Титульная страница}
\AbstractText{Аннотация}
\AcknowledgmentsText{Благодарности}
\AuthorText{Автор}
\Author{Фамилия Имя Отчество}
\AuthorEng{Author Name}
\Advisor{Фамилия И.О.}
\AdvisorDegree{}
\ListingText{Листинг}
\AlgorithmText{Алгоритм}

\renewcommand{\singlespace}{\setstretch{1}}
\newcommand{\oneandhalfspace}{\setstretch{1.33}}
\renewcommand{\doublespace}{\setstretch{1.66}}
\newcommand{\fixmargins}{}

\renewcommand{\contentsname}{Содержание}
\renewcommand{\refname}{Список литературы}

\newcommand{\msu@maybeLogo}{%
  \IfFileExists{img/gzlogo.png}{\includegraphics[height=35mm,keepaspectratio]{img/gzlogo.png}\\[2mm]}{%
    \IfFileExists{gzlogo.png}{\includegraphics[height=35mm,keepaspectratio]{gzlogo.png}\\[2mm]}{\vspace*{35mm}}%
  }%
}

\newcommand{\msu@titlelines}{%
  \textbf{\@TitleTop}\\
  \ifdefined\@TitleMiddle\textbf{\@TitleMiddle}\\\fi
  \ifdefined\@TitleBottom\textbf{\@TitleBottom}\\\fi
  \ifdefined\@TitleTopEng\vspace{0.5\baselineskip}\textbf{\@TitleTopEng}\\\fi
  \ifdefined\@TitleMiddleEng\textbf{\@TitleMiddleEng}\\\fi
  \ifdefined\@TitleBottomEng\textbf{\@TitleBottomEng}\\\fi
}

\newcommand{\makemsutitlepage}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \pdfbookmark[1]{\@TitlePageText}{bm:title}
  \begin{center}
    \msu@maybeLogo
    \@University\\
    \@Faculty\\
    \@Chair\\
    \ifdefined\@Lab {\scriptsize \@Lab\\}\fi
    \vspace{22mm}
    {\large \@Author\\[\baselineskip]}
    {\large \msu@titlelines}
    \vspace{16mm}
    \@docname\\
    \vfill
    \begin{flushright}
      \textbf{\@AdvisorText:}\\
      {\large \@AdvisorDegree~\@Advisor\\}
      \ifdefined\@Consultant
        \vspace{0.7\baselineskip}
        \textbf{\@ConsultantText:}\\
        {\large \ifdefined\@ConsultantDegree\@ConsultantDegree\fi~\@Consultant\\}
      \fi
    \end{flushright}
    \vfill
    \@City, \@Year
  \end{center}
  \end{titlepage}
}

\newcommand{\abstractpage}{%
  \ifdefined\@Abstract
    \clearpage
    \pdfbookmark[1]{\@AbstractText}{bm:abstract}
    \section*{\centering \@AbstractText}
    \vspace{0.75\baselineskip}
    \oneandhalfspace
    \@Abstract
    \clearpage
  \fi
  \ifdefined\@AbstractEng
    \clearpage
    \pdfbookmark[1]{Abstract}{bm:abstract-en}
    \section*{\centering Abstract}
    \begin{center}
      \ifdefined\@TitleTopEng\@TitleTopEng\else\@TitleTop\fi
      \ifdefined\@TitleMiddleEng\\\@TitleMiddleEng\fi
      \ifdefined\@TitleBottomEng\\\@TitleBottomEng\fi\\[\baselineskip]
      \textit{\@AuthorEng}
    \end{center}
    \oneandhalfspace
    \@AbstractEng
    \clearpage
  \fi
}

\newcommand{\acknowledgmentspage}{%
  \ifdefined\@Acknowledgments
    \clearpage
    \pdfbookmark[1]{\@AcknowledgmentsText}{bm:acknowledgments}
    \section*{\centering \@AcknowledgmentsText}
    \oneandhalfspace
    \@Acknowledgments
    \clearpage
  \fi
}

\newcommand{\makepreliminarypages}{%
  \makemsutitlepage
  \abstractpage
  \acknowledgmentspage
}

\newcommand{\PDFTitle}{\@Author~-~\@TitleTop\ifdefined\@TitleMiddle~\@TitleMiddle\fi\ifdefined\@TitleBottom~\@TitleBottom\fi}
\newcommand{\PDFAuthor}{\@Author}

\makeatother

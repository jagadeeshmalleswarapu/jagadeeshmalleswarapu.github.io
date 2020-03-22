$(document).ready(function(){
	// alert('working');
	$( function() {
		var data = [
		  "Action",
          "Adventure",
          "Animation",
          "Children",
          "Comedy",
          "Crime",
          "Documentary",
          "Drama",
          "Fantasy",
          "Film-Noir",
          "Horror",
          "Musical",
          "Mystery",
          "Romance",
          "Sci-Fi",
          "Thriller",
          "War",
          "Western",
          "(no genres listed)"
		]

	

	$( ".genress" ).autocomplete({
		source: data
	});
	});
});